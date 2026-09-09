'''
Attention transport module for AT-LG-ODE.

Implements the "Transport the evidence" step of the AT-LG-ODE design:

  r_ij(t) = sum_{s: obj(s)=j, obj(t')=i} alpha_{s->t'} * K_psi(t - t_s),   K_psi(dt) = exp(-lambda*dt)

  w_ij(t) = A_ij * r_ij(t) / (sum_k A_ik * r_ik(t) + eps)

r_ij(t) is only ever evaluated causally (t >= t_s); attention evidence from a source
observation in the future of the current ODE time contributes nothing.

The module is stateful across one forward/solve call: `build_cache` is called once right
after the encoder runs (attention doesn't depend on which trajectory sample or ODE time we're
at), `set_adjacency` is called once before the ODE solve starts, and `compute_weights(t)` is
then called by the ODE function at every solver evaluation.
'''
import torch
import torch.nn as nn


class AttentionTransport(nn.Module):

    def __init__(self, num_atoms, lam_init=5.0, eps=1e-6, learnable_lambda=True):
        super(AttentionTransport, self).__init__()
        self.num_atoms = num_atoms
        self.eps = eps

        log_lam_init = torch.log(torch.tensor(float(lam_init)))
        if learnable_lambda:
            self.log_lambda = nn.Parameter(log_lam_init)
        else:
            self.register_buffer('log_lambda', log_lam_init)

        pairs = [(i, j) for i in range(num_atoms) for j in range(num_atoms) if j != i]
        pair_i = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        pair_j = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        self.register_buffer('pair_i', pair_i)
        self.register_buffer('pair_j', pair_j)

        self._cache = None
        self._adjacency = None
        self._n_traj_samples = 1

    @property
    def lam(self):
        return torch.exp(self.log_lambda)

    def reset(self):
        self._cache = None
        self._adjacency = None
        self._n_traj_samples = 1

    def build_cache(self, attention, edge_index, object_id, t_s, node_batch):
        '''
        attention:  [E] captured post-softmax attention from the encoder's last GTrans layer
        edge_index: [2, E] (sender, receiver) node indices, column-aligned with `attention`
        object_id:  [num_nodes] physical object id (0..num_atoms-1) of every encoder node
        t_s:        [num_nodes] decoder-frame-aligned absolute time of every encoder node
        node_batch: [num_nodes] which sample in the batch (0..B-1) every encoder node belongs to
        '''
        sender = edge_index[0]
        receiver = edge_index[1]

        obj_j = object_id[sender]
        obj_i = object_id[receiver]
        graph_id = node_batch[sender]

        inter_object = obj_i != obj_j

        # Detached on purpose: gradients into the encoder's relational attention already flow
        # through the normal z0 (attention-pooling) path used by Stage 4, unchanged from
        # LG-ODE. Keeping this cache out of the graph lets the ODE solve use the memory-
        # efficient adjoint method (backprop through hundreds of solver evaluations while
        # holding this cache live would otherwise force it to retain the full unrolled graph).
        self._cache = {
            'graph': graph_id[inter_object],
            'i': obj_i[inter_object],
            'j': obj_j[inter_object],
            'ts': t_s[sender][inter_object].detach(),
            'alpha': attention[inter_object].detach(),
        }

    def has_cache(self):
        return self._cache is not None and self._cache['alpha'].numel() > 0

    def set_adjacency(self, adjacency, n_traj_samples=1):
        '''
        adjacency: [B, num_atoms*(num_atoms-1)] physical edge existence (0/1), in the same
                   (receiver-outer, sender-inner) order as rel_rec/rel_send/graph_augmented.
        '''
        self._adjacency = adjacency.float()
        self._n_traj_samples = n_traj_samples

    def compute_weights(self, t_query):
        '''
        Returns edge weights w_ij(t) of shape [B*n_traj_samples, num_atoms*(num_atoms-1)],
        matching the batch layout of `all_msgs` inside the ODE function (trajectory-sample
        outer, batch-sample inner -- see diffeq_solver.forward's `graph_augmented`).
        '''
        assert self._adjacency is not None, "call set_adjacency() before compute_weights()"
        B = self._adjacency.size(0)
        N = self.num_atoms
        device = self._adjacency.device

        if not self.has_cache():
            w = torch.zeros(B, N * (N - 1), device=device)
        else:
            c = self._cache
            t_query = t_query.reshape(())
            delta = t_query - c['ts']
            causal = (delta >= 0).to(delta.dtype)
            decay = torch.exp(-self.lam * torch.clamp(delta, min=0.0))
            weighted = c['alpha'] * decay * causal

            flat_idx = (c['graph'] * N + c['i']) * N + c['j']
            r_flat = torch.zeros(B * N * N, device=device, dtype=weighted.dtype)
            r_flat = r_flat.index_add(0, flat_idx, weighted)
            r = r_flat.view(B, N, N)

            r_offdiag = r[:, self.pair_i, self.pair_j]  # [B, N*(N-1)]

            Ar = self._adjacency * r_offdiag
            Ar_grid = Ar.view(B, N, N - 1)
            denom = Ar_grid.sum(dim=2, keepdim=True) + self.eps
            w = (Ar_grid / denom).view(B, N * (N - 1))

        if self._n_traj_samples > 1:
            w = w.repeat(self._n_traj_samples, 1)
        return w

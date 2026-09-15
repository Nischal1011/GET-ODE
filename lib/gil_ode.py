'''
GIL-ODE (Graph Innovation-Lifting ODE): a differentiable, mask-conditioned layer that lifts
observation corrections from observed nodes to unobserved nodes via a graph-regularized,
closed-form linear solve, layered on top of a continuous local+residual-graph latent ODE.

Differs from every other model in this project:
- LG-ODE infers z0 once from the whole context, then evolves/decodes -- no per-observation
  correction during the rollout.
- ODE-RNN (lib/baseline_odernn.py) corrects ONLY the observed node at each event, independently
  per node -- an unobserved node's state is only ever touched by its own local ODE.
- Edge-GNN/RNN-NRI use graph structure in the encoder or a discrete rollout, not a continuous,
  per-event joint correction step.
GIL-ODE's continuous dynamics already mix nodes (a residual graph term in the vector field,
gated by an alpha warm-started slightly positive so training starts close to, but not exactly
at, independent per-node dynamics -- see CHANGES.md for why a hard-zero init stalled), and its
correction step explicitly lifts single-node innovations to the whole graph
via a Laplacian-regularized least-squares solve, rather than an unrestricted learned message
(a Graph GRU) or no cross-node correction at all (ODE-RNN).

Documented simplifications (flagged, not silent):
- The edge feature e_ij (kept separate from the relation feature c_ij in the spec) is dropped;
  phi_theta and g_theta take (h_i, h_j, c_ij) only. No natural, dataset-agnostic e_ij was
  specified beyond what's already covered by S_ij/c_ij.
- The gate beta uses (time since last observation, this event's observed fraction, ||delta_i||)
  -- "predicted uncertainty" is omitted since this is a deterministic latent state with no
  tracked variance (matching ODE-RNN/Latent-ODE's own choice not to track one either, except
  Latent-ODE's z0 itself).
- lambda and rho (Laplacian-regularization and unobserved-node dampening strength) are
  learnable, softplus-parameterized scalars rather than fixed hyperparameters; epsilon is a
  small fixed constant purely for solve stability.

Dataset-specific support (S) and relation (c) construction (lib/gil_dataset.py):
- Springs: S_ij = the dataset's own sparse physical graph (0/1); c_ij = 0 (no signed relation).
- Charged: S_ij = 1 for all i != j (every pair interacts, per spec); c_ij = the raw +-1
  charge-product sign, recovered from the dataloader's 0/1-cast relation label.
- IEEE39-Gen: S_ij = 1 for all i != j (complete interaction-support assumption); c_ij = 0. Never
  read w_ij as recovered transmission-line topology (see reports/DATA_CHARACTERISTICS.md).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import lib.utils as utils


class GILODEFunc(nn.Module):
    '''dh_i/dt = f_local(h_i) + alpha * f_graph(h_i, {h_j}); S/c are set once per batch/solve.'''

    def __init__(self, hidden_dim, rel_dim=1):
        super(GILODEFunc, self).__init__()
        self.f_local = utils.create_net(hidden_dim, hidden_dim, n_layers=1, n_units=hidden_dim, nonlinear=nn.Tanh)
        # n_layers=2 (was 1): once alpha actually engages (see the v1->v2 fix in CHANGES.md), the
        # graph term's own capacity became the limiting factor on springs/charged-extrap -- alpha
        # grew but MSE didn't move, pointing at phi itself rather than the gate.
        self.phi = utils.create_net(hidden_dim * 2 + rel_dim, hidden_dim, n_layers=2, n_units=hidden_dim, nonlinear=nn.Tanh)
        # Warm-started slightly positive (not exactly 0): a hard-zero init left no gradient
        # incentive to grow alpha under long extrapolation horizons (see CHANGES.md), so training
        # settled back near 0 instead of learning the graph term where it mattered.
        self.alpha = nn.Parameter(torch.tensor(0.15))
        utils.init_network_weights(self.f_local)
        utils.init_network_weights(self.phi)
        self.S = None
        self.c = None

    def set_graph(self, S, c):
        self.S = S  # [B, N, N]
        self.c = c  # [B, N, N]

    def forward(self, t, h):
        B, N, H = h.shape
        local = self.f_local(h)

        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)
        c_ij = self.c.unsqueeze(-1)
        edge_in = torch.cat([h_i, h_j, c_ij], dim=-1)
        phi_out = self.phi(edge_in)  # [B, N, N, H], phi_out[:, i, j] = phi(h_i, h_j, c_ij)

        S_ = self.S.unsqueeze(-1)
        deg = self.S.sum(dim=2, keepdim=True).clamp(min=1.0)  # [B, N, 1]
        graph_term = (S_ * phi_out).sum(dim=2) / deg  # sum over j -> [B, N, H]

        return local + self.alpha * graph_term


class GraphLifting(nn.Module):
    '''Learned conductance w_ij, weighted Laplacian, and the closed-form innovation-lifting solve
    delta* = (M + lambda*L_w + rho*(I-M) + eps*I)^-1 @ (M @ r).'''

    def __init__(self, hidden_dim, rel_dim=1):
        super(GraphLifting, self).__init__()
        self.g = utils.create_net(hidden_dim * 2 + rel_dim, 1, n_layers=1, n_units=hidden_dim, nonlinear=nn.Tanh)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.log_rho = nn.Parameter(torch.tensor(0.0))
        self.eps = 1e-4
        utils.init_network_weights(self.g)

    def forward(self, h_minus, S, c, r, mask):
        '''
        h_minus: [B, N, H] (pre-correction state), S/c: [B, N, N], r: [B, N, H] (innovation,
        zero where unobserved), mask: [B, N] float (1 where observed at this event).
        Returns delta [B, N, H].
        '''
        B, N, H = h_minus.shape
        h_i = h_minus.unsqueeze(2).expand(B, N, N, H)
        h_j = h_minus.unsqueeze(1).expand(B, N, N, H)
        c_ij = c.unsqueeze(-1)
        edge_in = torch.cat([h_i, h_j, c_ij], dim=-1)
        w = S * F.softplus(self.g(edge_in).squeeze(-1))  # [B, N, N]
        w = 0.5 * (w + w.transpose(1, 2))  # symmetrize: g_theta(h_i,h_j) need not equal g_theta(h_j,h_i)

        D = torch.diag_embed(w.sum(dim=2))
        L = D - w

        lam = F.softplus(self.log_lambda)
        rho = F.softplus(self.log_rho)

        eye = torch.eye(N, device=h_minus.device).unsqueeze(0)
        M_diag = torch.diag_embed(mask)
        A = M_diag + lam * L + rho * torch.diag_embed(1 - mask) + self.eps * eye

        rhs = mask.unsqueeze(-1) * r
        delta = torch.linalg.solve(A, rhs)
        return delta


class GateNet(nn.Module):
    '''beta = sigmoid(MLP([time since last obs, this event's observed fraction, ||delta_i||])).'''

    def __init__(self):
        super(GateNet, self).__init__()
        self.net = utils.create_net(3, 1, n_layers=1, n_units=32, nonlinear=nn.ReLU)
        utils.init_network_weights(self.net)

    def forward(self, time_since_obs, obs_density, delta_norm):
        inp = torch.stack([time_since_obs, obs_density, delta_norm], dim=-1)  # [B, N, 3]
        return torch.sigmoid(self.net(inp))  # [B, N, 1]


class GILODEModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_atoms, device):
        super(GILODEModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.device = device

        self.encoder_proj = nn.Linear(input_dim, hidden_dim)  # E_theta
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.ode_func = GILODEFunc(hidden_dim)
        self.lifting = GraphLifting(hidden_dim)
        self.gate = GateNet()
        utils.init_network_weights(self.encoder_proj)
        utils.init_network_weights(self.decoder)

    def forward(self, dense, obs_mask, grid_times, decoder_time_steps, S, c):
        '''
        dense: [B, N, T, D], obs_mask: [B, N, T] bool, grid_times: [T] (union of every node's
        observation times and the decoder's query times, ascending), S/c: [B, N, N].
        Returns pred_x [B, N, T_Q, D] at decoder_time_steps.
        '''
        B, N, T, D = dense.shape
        self.ode_func.set_graph(S, c)

        h = torch.zeros(B, N, self.hidden_dim, device=self.device)
        time_since_obs = torch.zeros(B, N, device=self.device)
        outputs = torch.zeros(B, N, T, self.hidden_dim, device=self.device)

        prev_t = grid_times[0]
        for i in range(T):
            cur_t = grid_times[i]
            if i > 0 and (cur_t - prev_t).abs() > 1e-8:
                t_span = torch.stack([prev_t, cur_t])
                h = odeint(self.ode_func, h, t_span, method='rk4')[-1]
                time_since_obs = time_since_obs + (cur_t - prev_t)

            mask_i = obs_mask[:, :, i]  # [B, N] bool
            if mask_i.any():
                mask_f = mask_i.float()
                y_i = dense[:, :, i, :]
                r = mask_f.unsqueeze(-1) * (self.encoder_proj(y_i) - h)
                delta = self.lifting(h, S, c, r, mask_f)
                obs_density = mask_f.mean(dim=1, keepdim=True).expand(-1, N)
                beta = self.gate(time_since_obs, obs_density, delta.norm(dim=-1))
                h = h + beta * delta
                time_since_obs = torch.where(mask_i, torch.zeros_like(time_since_obs), time_since_obs)

            outputs[:, :, i] = h
            prev_t = cur_t

        query_idx = torch.searchsorted(grid_times, decoder_time_steps)
        h_at_query = outputs[:, :, query_idx]
        return self.decoder(h_at_query)

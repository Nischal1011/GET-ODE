'''
Structured GIL-ODE (Graph Innovation-Lifting ODE): a differentiable, mask-conditioned layer that
lifts observation corrections from observed nodes to unobserved nodes via a graph-regularized,
closed-form linear solve, layered on top of a continuous local+residual-graph latent ODE.

Differs from every other model in this project:
- LG-ODE infers z0 once from the whole context, then evolves/decodes -- no per-observation
  correction during the rollout.
- ODE-RNN (lib/baseline_odernn.py) corrects ONLY the observed node at each event, independently
  per node -- an unobserved node's state is only ever touched by its own local ODE.
- Edge-GNN/RNN-NRI use graph structure in the encoder or a discrete rollout, not a continuous,
  per-event joint correction step.
GIL-ODE's continuous dynamics already mix nodes (a residual graph term in the vector field), and
its correction step explicitly lifts single-node innovations to the whole graph via a
Laplacian-regularized least-squares solve, rather than an unrestricted learned message (a Graph
GRU) or no cross-node correction at all (ODE-RNN).

This is the second major revision (the first, Experiment 1 in CHANGES.md, tested relation-expert
+ sum aggregation in isolation and found it insufficient -- helped springs/IEEE39 modestly, hurt
charged). This revision combines three targeted changes rather than one, following a diagnosis
of *why* the vector field and lifting steps were each still limiting performance:

1. **Hard-anchored innovation lifting** (GraphLifting): an observed node's correction was
   previously smoothed by the Laplacian and reduced by the confidence gate even though it has an
   exact, trusted observation available -- there is no measurement-noise model in these
   datasets, so a directly observed node should be anchored to its encoded observation exactly,
   and only *unobserved* nodes should have their correction inferred through the graph. Rows of
   the linear system corresponding to observed nodes are now hard-replaced with the identity
   (delta_i = r_i exactly); unobserved rows are unchanged (still solved via the weighted
   Laplacian, still coupled to observed nodes' now-exact corrections through the off-diagonal
   terms). The confidence gate (GateNet) is applied only to unobserved nodes' inferred
   corrections, not to observed nodes' exact ones -- see GILODEModel.forward.
2. **Relation-expert, degree-aware messages** (GILODEFunc): one relation-expert MLP per sign of
   c_ij (phi_pos/phi_neg, matching NRI/LG-ODE's own per-relation-type message design) replaces
   the single shared MLP; the message now also takes the pairwise difference h_j - h_i (a
   translation-invariant relative feature), not just the raw concatenation. Aggregation combines
   both sum (physical force is additive over neighbors) and mean (normalized collective effect,
   useful for dense/complete graphs) via a small learned combiner net, rather than committing to
   one aggregation rule -- Experiment 1's plain sum() helped springs (variable degree) but hurt
   charged (complete graph, where an unmodulated ~4x scale-up destabilized training).
3. **State-dependent graph gate** (GILODEFunc): the single global scalar alpha -- shared across
   every node, channel, timestep, and trajectory -- is replaced with a bounded, per-node gate
   computed from the node's own state, its aggregated incoming message, and its degree,
   warm-started near 0.1 (matching alpha's own earlier warm-start rationale) via a biased final
   layer. This directly targets the fragility already observed with a single scalar: boosting
   alpha's learning rate helped IEEE39 but overshot on charged, because one number was being
   asked to be simultaneously right for every node/state/trajectory in every dataset.

Documented simplifications (flagged, not silent):
- The edge feature e_ij (kept separate from the relation feature c_ij in the original spec) is
  still dropped; the pairwise difference h_j - h_i is used instead as the relative feature.
- The gate beta (for unobserved nodes' inferred corrections) uses (time since last observation,
  this event's observed fraction, ||delta_i||) -- "predicted uncertainty" is omitted since this
  is a deterministic latent state with no tracked variance.
- lambda and rho (Laplacian-regularization and unobserved-node dampening strength) are
  learnable, softplus-parameterized scalars rather than fixed hyperparameters; epsilon is a
  small fixed constant purely for solve stability.
- The state-dependent gate is per-node (a scalar), not per-channel, to avoid adding a second
  axis of complexity on top of the first revision's single-scalar fix without evidence it's
  needed.

Dataset-specific support (S) and relation (c) construction (lib/gil_dataset.py):
- Springs: S_ij = the dataset's own sparse physical graph (0/1); c_ij = 0 (no signed relation).
- Charged: S_ij = 1 for all i != j (every pair interacts, per spec); c_ij = the raw +-1
  charge-product sign, recovered from the dataloader's 0/1-cast relation label.
- IEEE39-Gen: S_ij = the dataset's own physical support graph (0/1), Kron-reduced from the real
  IEEE 39-bus network's admittance matrix (data/build_ieee39_kron_graph.py), not a complete-
  graph placeholder (was, before CHANGES.md's Part on the IEEE39 physical graph); c_ij = 0.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import lib.utils as utils


class GILODEFunc(nn.Module):
    '''
    dh_i/dt = f_local(h_i) + g_i(t) * f_interaction(h_i, m_i); S/c are set once per batch/solve.
    g_i(t) is a per-node, state-dependent gate (was a single global scalar alpha) and m_i is a
    learned sum+mean combination of relation-expert messages (was a single mean-aggregated MLP).
    See this file's module docstring for the full rationale.
    '''

    def __init__(self, hidden_dim):
        super(GILODEFunc, self).__init__()
        self.f_local = utils.create_net(hidden_dim, hidden_dim, n_layers=1, n_units=hidden_dim, nonlinear=nn.Tanh)
        # Relation-expert messages, now including the pairwise difference (h_j - h_i).
        self.phi_pos = utils.create_net(hidden_dim * 3, hidden_dim, n_layers=2, n_units=hidden_dim, nonlinear=nn.Tanh)
        self.phi_neg = utils.create_net(hidden_dim * 3, hidden_dim, n_layers=2, n_units=hidden_dim, nonlinear=nn.Tanh)
        # Learned combination of sum- and mean-aggregated messages, plus log(1+degree).
        self.psi = utils.create_net(hidden_dim * 2 + 1, hidden_dim, n_layers=1, n_units=hidden_dim, nonlinear=nn.Tanh)
        # f_interaction(h_i, m_i) -- the gated term added to the local dynamics.
        self.f_interaction = utils.create_net(hidden_dim * 2, hidden_dim, n_layers=1, n_units=hidden_dim, nonlinear=nn.Tanh)
        # State-dependent gate: sigmoid(a_theta([h_i, m_i, log(1+deg_i)]) + b0), b0 chosen so the
        # gate starts near 0.1 -- same warm-start rationale as the scalar alpha it replaces
        # (a hard-zero start left no gradient incentive to grow under long horizons).
        self.gate_net = utils.create_net(hidden_dim * 2 + 1, 1, n_layers=1, n_units=hidden_dim, nonlinear=nn.Tanh)

        utils.init_network_weights(self.f_local)
        utils.init_network_weights(self.phi_pos)
        utils.init_network_weights(self.phi_neg)
        utils.init_network_weights(self.psi)
        utils.init_network_weights(self.f_interaction)
        utils.init_network_weights(self.gate_net)
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(-2.1972)  # logit(0.1)

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
        edge_in = torch.cat([h_i, h_j, h_j - h_i], dim=-1)

        is_pos = (self.c > 0).unsqueeze(-1)
        msg = torch.where(is_pos, self.phi_pos(edge_in), self.phi_neg(edge_in))  # [B, N, N, H]

        S_ = self.S.unsqueeze(-1)
        deg = self.S.sum(dim=2, keepdim=True)  # [B, N, 1]
        m_sum = (S_ * msg).sum(dim=2)  # [B, N, H]
        m_mean = m_sum / deg.clamp(min=1.0)
        log_deg = torch.log1p(deg)  # [B, N, 1]
        m_tilde = self.psi(torch.cat([m_sum, m_mean, log_deg], dim=-1))  # [B, N, H]

        gate_in = torch.cat([h, m_tilde, log_deg], dim=-1)
        g = torch.sigmoid(self.gate_net(gate_in))  # [B, N, 1]
        interaction = self.f_interaction(torch.cat([h, m_tilde], dim=-1))  # [B, N, H]

        return local + g * interaction


class GraphLifting(nn.Module):
    '''
    Learned conductance w_ij, weighted Laplacian, and the innovation-lifting solve -- now
    hard-anchored (see module docstring, point 1): rows for observed nodes are the identity
    exactly (delta_i = r_i, no Laplacian smoothing, no leakage from other nodes' corrections);
    rows for unobserved nodes keep the original graph-regularized equation, still coupled to
    observed nodes' now-exact corrections through the off-diagonal terms.
    '''

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
        Returns delta [B, N, H], with delta_i == r_i exactly wherever mask_i == 1.
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

        eye = torch.eye(N, device=h_minus.device).unsqueeze(0).expand(B, -1, -1)
        A_unobserved = lam * L + rho * torch.diag_embed(1 - mask) + self.eps * eye
        is_observed_row = mask.unsqueeze(-1).expand(-1, -1, N).bool()  # True across an observed node's whole row
        A = torch.where(is_observed_row, eye, A_unobserved)

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
                # Observed nodes get their exact correction unconditionally (delta_i == r_i,
                # hard-anchored in GraphLifting); the gate only weighs *inferred* corrections
                # for unobserved nodes -- see module docstring, point 1.
                mask_col = mask_f.unsqueeze(-1)
                h = h + mask_col * delta + (1 - mask_col) * beta * delta
                time_since_obs = torch.where(mask_i, torch.zeros_like(time_since_obs), time_since_obs)

            outputs[:, :, i] = h
            prev_t = cur_t

        query_idx = torch.searchsorted(grid_times, decoder_time_steps)
        h_at_query = outputs[:, :, query_idx]
        return self.decoder(h_at_query)

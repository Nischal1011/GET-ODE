'''
CSG-ODE: ControlSynth Graph ODE (Wang, Wang & Liang, ICML 2025, PMLR 267) -- reimplementation.

No official code has been released (checked GitHub, the OpenReview submission and the ICML page,
2026-10-06). This file reimplements the method from the paper's equations and Appendix, with the
paper's own hyperparameters (Table 8). The decoder's numerical scheme follows the authors' cited
ControlSynth Neural ODE reference code (Mei, Zheng & Li, NeurIPS 2024;
github.com/ContinuumCoder/ControlSynth-Neural-ODE), whose `DynamicODESolver` integrates state and
control jointly with explicit Euler, one step per time interval, initialising the control from the
state (u0 = y0).

Architecture (paper section and equation):

ENCODER -- "Latent Distribution Generation" (Sec. 3.1)
  * Adaptive graph   G = softmax(relu(Es Et^T))                                          (Eq. 1)
  * Edge importance from total communicability, via the Frechet derivative of the matrix
    exponential estimated by a central difference with beta = 2/N * 1e-4:
        Lf = [exp0(Go^T + b ee^T) - exp0(Go^T - b ee^T)] / 2b,   D = Go (.) Lf / ||Lf||_F   (Eqs. 2-3)
    Computed in float64: with beta ~ 1e-5 the difference cancels catastrophically in float32.
  * Mixed graph      Gmix = G + W1 (.) D                                                  (Eq. 4)
  * Per-timestamp snapshot, density-aware:
        G^t_ij = Gmix_ij * M^t_ij * (1 - W2_ij * alpha * |sigma(R^t_i) - sigma(R^t_j)|)    (Eqs. 5-6)
    with R the sampling density of Appendix A and M^t the both-observed mask.
  * Observation embedding E = MLP_E(x) (Eq. 8); node embedding Q_i = phi(mean of node i's
    observed E) (Eqs. 9-10); node-specific graph-convolution weights Theta = Q W_C (Eq. 11).
  * GCRNN update (Eq. 7): an AGCRN-style graph-convolutional GRU (Bai et al. 2020, which the paper
    follows for Eqs. 9-11), two supports [I, G^t]. A node's hidden state is updated only at its own
    observation times, so H_i is "the hidden state at node i's last observation" as Eq. (12) reads.
  * Posterior q(z0_i) = N(mu_i, sigma_i), (mu_i, sigma_i) = phi(H_i)                      (Eqs. 12-13)

DECODER -- "Latent State Generation" (Sec. 3.2)
        dz/dt = A0 z + sum_{j=1..M} A_j f_j(MLP_j(z)) + g(c)                               (Eq. 14)
        dc/dt = GNN(z_1, ..., z_N)
  M = 2 subnetworks, width 128, depth 1, f_j = tanh, so MLP_j: d -> 128 and A_j: 128 -> d;
  g is linear (as the reference ODEFuncG); the control GNN is NRI-style message passing over the
  dataset graph. Latent 16 + augment 64, as in LG-ODE and Table 8. Solved with explicit Euler
  (Table 8: "ODEsolve: Euler"). Decoder p(x|z) is linear, as in LG-ODE.

TRAINING -- ELBO (Eq. 17), through VAE_Baseline.compute_all_losses: the same likelihood, KL and
KL warm-up as Corrected LG-ODE in this project.

Unspecified in the paper, resolved here (documented, not silent):
  * dimension of Es/Et: d1 = input feature dimension, following the paper's notation in Sec. 2;
  * W1, W2 initialised to ones; sigma in Eq. (5) is the logistic sigmoid;
  * dropout 0.2 (Table 8) applied inside MLP_E;
  * control state initialised from the latent state, c0 = z0 (reference code: u0 = y0).
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import lib.utils as utils
from lib.base_models import VAE_Baseline
from lib.gil_dataset import prepare_gil_batch


def init_weights(net, std=0.1):
    '''Same scheme as lib/utils.init_network_weights (N(0, 0.1) weights, zero bias), but tolerating
    bias-free layers: A0 and A_j in Eq. (14) are pure matrices. The shared helper is left untouched
    because every other model in the project relies on it.'''
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def edge_importance(Go):
    '''Eqs. (2)-(3), batched. Go: [B, N, N]. Returns D: [B, N, N]. float64 internally.'''
    B, N, _ = Go.shape
    beta = 2.0 / N * 1e-4
    G64 = Go.double().transpose(1, 2)
    J = torch.ones(N, N, dtype=torch.float64, device=Go.device)
    # exp0(X) = expm(X) - I; the identities cancel in the difference, so expm alone suffices.
    Lf = (torch.linalg.matrix_exp(G64 + beta * J) - torch.linalg.matrix_exp(G64 - beta * J)) / (2 * beta)
    nrm = Lf.flatten(1).norm(dim=1).clamp(min=1e-12).view(B, 1, 1)
    return (Go.double() * Lf / nrm).to(Go.dtype)


def sampling_density(mask, grid_times):
    '''Appendix A. mask: [B, N, T] bool, grid_times: [T]. Returns R: [B, N, T], defined where observed.'''
    B, N, T = mask.shape
    t = grid_times.view(1, 1, T).expand(B, N, T)
    inf = torch.full_like(t, float('inf'))
    obs_t = torch.where(mask, t, inf)
    # previous / next observation time of the same node, strictly before / after each step
    prev = torch.full_like(t, -float('inf')); nxt = inf.clone()
    run = torch.full((B, N), -float('inf'), device=t.device, dtype=t.dtype)
    for i in range(T):
        prev[:, :, i] = run
        run = torch.where(mask[:, :, i], t[:, :, i], run)
    run = torch.full((B, N), float('inf'), device=t.device, dtype=t.dtype)
    for i in reversed(range(T)):
        nxt[:, :, i] = run
        run = torch.where(mask[:, :, i], t[:, :, i], run)
    has_p, has_n = torch.isfinite(prev), torch.isfinite(nxt)
    span = (torch.where(mask, t, -inf).amax(-1) - torch.where(mask, t, inf).amin(-1)).clamp(min=0)
    R = torch.where(has_p & has_n, ((t - prev) + (nxt - t)) / 2,
        torch.where(has_p, t - prev,
        torch.where(has_n, nxt - t, (span / 2).unsqueeze(-1).expand(B, N, T))))
    return torch.where(mask, R, torch.zeros_like(R))


class NodeAdaptiveGraphConv(nn.Module):
    '''AGCRN graph convolution with node-specific weights Theta_i = Q_i W_C (Eq. 11); supports [I, A].'''
    def __init__(self, q, c_in, c_out, k_supports=2):
        super().__init__()
        self.w_pool = nn.Parameter(torch.empty(q, k_supports, c_in, c_out))
        self.b_pool = nn.Parameter(torch.zeros(q, c_out))
        nn.init.xavier_normal_(self.w_pool)

    def node_weights(self, Q):
        '''Theta = Q W_C (Eq. 11). Q is fixed over the sequence, so this is computed ONCE per forward
        pass and reused at every timestamp. Recomputing it inside the time loop (as an earlier version
        did) is mathematically identical but keeps a [B, N, K, C, O] copy alive for backward at every
        step, which filled 32 GB of GPU memory on IEEE39.'''
        return torch.einsum('bnq,qkio->bnkio', Q, self.w_pool), Q @ self.b_pool

    def forward(self, x, A, W, b):
        # x: [B, N, C], A: [B, N, N], W: [B, N, K, C, O], b: [B, N, O]
        xg = torch.stack([x, A @ x], dim=2)                               # [B, N, K, C]
        return torch.einsum('bnki,bnkio->bno', xg, W) + b


class GCRNNCell(nn.Module):
    def __init__(self, q, c_in, hidden):
        super().__init__()
        self.hidden = hidden
        self.gate = NodeAdaptiveGraphConv(q, c_in + hidden, 2 * hidden)
        self.cand = NodeAdaptiveGraphConv(q, c_in + hidden, hidden)

    def node_weights(self, Q):
        return self.gate.node_weights(Q), self.cand.node_weights(Q)

    def forward(self, x, h, A, weights):
        (Wg, bg), (Wc, bc) = weights
        zr = torch.sigmoid(self.gate(torch.cat([x, h], -1), A, Wg, bg))
        z, r = zr.split(self.hidden, dim=-1)
        hc = torch.tanh(self.cand(torch.cat([x, r * h], -1), A, Wc, bc))
        return z * h + (1 - z) * hc


class CSODEFunc(nn.Module):
    '''Eq. (14): dz = A0 z + sum_j A_j tanh(MLP_j(z)) + g(c);  dc = GNN(z).'''
    def __init__(self, d, n_sub=2, width=128):
        super().__init__()
        self.A0 = nn.Linear(d, d, bias=False)
        self.subnets = nn.ModuleList([nn.Linear(d, width) for _ in range(n_sub)])  # depth 1
        self.A = nn.ModuleList([nn.Linear(width, d, bias=False) for _ in range(n_sub)])
        self.g = nn.Linear(d, d)
        self.msg = nn.Sequential(nn.Linear(2 * d, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.node = nn.Linear(width, d)
        for m in [self.A0, self.g, self.node, *self.subnets, *self.A]:
            init_weights(m)
        init_weights(self.msg)

    def dz(self, z, c):
        out = self.A0(z) + self.g(c)
        for mlp, A in zip(self.subnets, self.A):
            out = out + A(torch.tanh(mlp(z)))
        return out

    def dc(self, z, S):
        # NRI-style message passing over the dataset graph S: [Bt, N, N]
        Bt, N, d = z.shape
        zi = z.unsqueeze(2).expand(Bt, N, N, d); zj = z.unsqueeze(1).expand(Bt, N, N, d)
        m = self.msg(torch.cat([zi, zj], -1))                              # [Bt, N, N, W]
        return self.node((S.unsqueeze(-1) * m).sum(2))


class CSGODE(VAE_Baseline):
    def __init__(self, input_dim, num_atoms, dataset, device, obsrv_std, z0_prior,
                 latent_dim=16, augment_dim=64, k=64, q=32, h=16, n_sub=2, sub_width=128,
                 alpha=0.5, dropout=0.2, solver='euler'):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, z0_prior=z0_prior,
                         device=device, obsrv_std=obsrv_std)
        self.N, self.dataset, self.alpha = num_atoms, dataset, alpha
        self.solver = solver   # 'euler' (paper, Table 8) or 'rk4' (same solver as LG-ODE in this project)
        self.latent_dim, self.augment_dim = latent_dim, augment_dim
        d = latent_dim + augment_dim
        N = num_atoms
        # Eq. (1)
        self.Es = nn.Parameter(torch.randn(N, input_dim)); self.Et = nn.Parameter(torch.randn(N, input_dim))
        # Eqs. (4)-(5)
        self.W1 = nn.Parameter(torch.ones(N, N)); self.W2 = nn.Parameter(torch.ones(N, N))
        # Eqs. (8)-(10)
        self.mlp_E = nn.Sequential(nn.Linear(input_dim, k), nn.ReLU(), nn.Dropout(dropout), nn.Linear(k, k))
        self.phi_Q = nn.Sequential(nn.Linear(k, q), nn.ReLU(), nn.Linear(q, q))
        # Eq. (7)
        self.cell = GCRNNCell(q, k, h)
        # Eq. (12)
        self.phi_post = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 * latent_dim))
        # Eq. (14) + decoder
        self.ode = CSODEFunc(d, n_sub=n_sub, width=sub_width)
        self.decoder = nn.Linear(d, input_dim)
        for m in [self.mlp_E, self.phi_Q, self.phi_post, self.decoder]:
            init_weights(m)

    # ---------------- encoder ----------------
    def encode(self, dense, mask, grid_times, S):
        B, N, T, D = dense.shape
        E = self.mlp_E(dense) * mask.unsqueeze(-1).float()                        # [B, N, T, k]
        cnt = mask.sum(-1, keepdim=True).clamp(min=1).float()
        Q = self.phi_Q(E.sum(2) / cnt)                                             # [B, N, q]
        G = torch.softmax(F.relu(self.Es @ self.Et.T), dim=1)                     # [N, N]
        Gmix = G.unsqueeze(0) + self.W1 * edge_importance(S)                       # [B, N, N]
        sR = torch.sigmoid(sampling_density(mask, grid_times))                     # [B, N, T]
        H = torch.zeros(B, N, self.cell.hidden, device=dense.device)
        weights = self.cell.node_weights(Q)                                         # once per forward
        for i in range(T):
            m = mask[:, :, i]
            if not m.any():
                continue
            mf = m.float()
            Mt = mf.unsqueeze(2) * mf.unsqueeze(1)
            dens = (sR[:, :, i].unsqueeze(2) - sR[:, :, i].unsqueeze(1)).abs()
            Gt = Gmix * Mt * (1 - self.W2 * self.alpha * dens)
            # Gradient checkpointing per timestep: identical gradients, but only the step inputs are kept
            # for backward instead of every intermediate (the full encoder held ~11 GB on IEEE39).
            Hn = (checkpoint(self.cell, E[:, :, i], H, Gt, weights, use_reentrant=False)
                  if torch.is_grad_enabled() else self.cell(E[:, :, i], H, Gt, weights))
            H = torch.where(m.unsqueeze(-1), Hn, H)                                # update observed nodes only
        post = self.phi_post(H)
        mu, std = post[..., :self.latent_dim], F.softplus(post[..., self.latent_dim:]) + 1e-4
        return mu.reshape(B * N, -1), std.reshape(B * N, -1)

    # ---------------- decoder ----------------
    def _euler_step(self, z, c, Sb, dt):
        return z + dt * self.ode.dz(z, c), c + dt * self.ode.dc(z, Sb)

    def _rk4_step(self, z, c, Sb, dt):
        '''Classical RK4 on the joint (z, c) system of Eq. (14). Same solver LG-ODE uses in this project,
        offered because the paper's own Table 9 shows CSG-ODE improves with a stronger solver than Euler.'''
        F = lambda z_, c_: (self.ode.dz(z_, c_), self.ode.dc(z_, Sb))
        k1z, k1c = F(z, c)
        k2z, k2c = F(z + dt / 2 * k1z, c + dt / 2 * k1c)
        k3z, k3c = F(z + dt / 2 * k2z, c + dt / 2 * k2c)
        k4z, k4c = F(z + dt * k3z, c + dt * k3c)
        return z + dt / 6 * (k1z + 2 * k2z + 2 * k3z + k4z), c + dt / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)

    def get_reconstruction(self, batch_en, batch_de, batch_g, n_traj_samples=1, run_backwards=True):
        dense, mask, grid_times, S, _ = prepare_gil_batch(batch_en, batch_de, batch_g, self.N, self.dataset, self.device)
        B, N = dense.shape[:2]
        mu, std = self.encode(dense, mask, grid_times, S)                          # [B*N, latent]
        z0 = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_traj_samples, *mu.shape, device=mu.device)
        z = torch.cat([z0, torch.zeros(n_traj_samples, B * N, self.augment_dim, device=mu.device)], -1)
        z = z.reshape(n_traj_samples * B, N, -1)
        Sb = S.repeat(n_traj_samples, 1, 1)

        tq = batch_de["time_steps"]
        pad = bool(tq[0] != 0)
        times = torch.cat([torch.zeros(1, device=tq.device, dtype=tq.dtype), tq]) if pad else tq
        c = z.clone()                                                              # c0 = z0 (reference: u0 = y0)
        outs = [z]
        for a, b in zip(times[:-1], times[1:]):                                    # explicit Euler, one step/interval
            dt = b - a
            # Checkpointed per Euler step for the same reason: the control GNN runs its message MLP over
            # every node pair for every trajectory sample (~18.7 GB held on IEEE39 without this).
            step = self._rk4_step if self.solver == 'rk4' else self._euler_step
            z, c = (checkpoint(step, z, c, Sb, dt, use_reentrant=False)
                    if torch.is_grad_enabled() else step(z, c, Sb, dt))
            outs.append(z)
        traj = torch.stack(outs, 2)                                                # [S*B, N, T', d]
        if pad:
            traj = traj[:, :, 1:]
        pred = self.decoder(traj).reshape(n_traj_samples, B * N, len(tq), -1)
        info = {"first_point": (mu.unsqueeze(0), std.unsqueeze(0), z0), "latent_traj": traj.detach()}
        return pred, info, None

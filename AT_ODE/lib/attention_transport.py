# # --- transport.py ------------------------------------------------------------
# from typing import Dict, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# TypePair = Tuple[int, int]  # (src_type, dst_type)
#
# class AttentionTransportPDE(torch.nn.Module):
#     """
#     Maintains and evolves lag histograms μ and a structural surrogate S for each type-pair.
#     All tensors are non-negative and row-normalized over K after updates.
#     """
#     def __init__(self, type_pairs, K: int, L: int,
#                  init_speed: float = 0.5,
#                  ema_kappa: float = 0.5,
#                  src_gain: float = 0.2,
#                  stride: int = 2,
#                  device: str = "cuda"):
#         super().__init__()
#         self.type_pairs = list(type_pairs)
#         self.K, self.L = K, L
#         self.ema_kappa = ema_kappa
#         self.src_gain = src_gain
#         self.stride = stride
#         self.register_buffer("_step_counter", torch.zeros((), dtype=torch.long))
#         # per type-pair transport speeds v_{cd} (positive, bounded by 1 via sigmoid)
#         self.v_raw = torch.nn.ParameterDict({
#             f"{c}_{d}": torch.nn.Parameter(torch.tensor(self._inv_sig(init_speed)))
#             for (c, d) in self.type_pairs
#         })
#         self._mu = {}   # (c,d) -> [B, Q, K, L]
#         self._S  = {}   # (c,d) -> [B, Q, K]
#         self.device = device
#
#     @staticmethod
#     def _sig(x): return torch.sigmoid(x)         # (0,1)
#     @staticmethod
#     def _inv_sig(y): return torch.log(y/(1-y))   # inverse for init
#
#     def reset_state(self):
#         self._mu.clear(); self._S.clear()
#         self._step_counter.zero_()
#
#     def _ensure_state(self, key: TypePair, shape_bqk: Tuple[int,int,int], device):
#         if key not in self._mu:
#             B, Q, K = shape_bqk
#             L = self.L
#             self._mu[key] = torch.zeros(B, Q, K, L, device=device)
#             self._S[key]  = torch.full((B, Q, K), 1.0 / K, device=device)  # uniform start
#
#     @torch.no_grad()
#     def transport_step(self,
#                        attn_event_bqkl: Dict[TypePair, torch.Tensor],
#                        dt: float):
#         """
#         attn_event_bqkl[(c,d)] : [B, Q, K, L], non-neg, row-normalized over K for each lag
#         Evolves μ and updates S via boundary injection.
#         """
#         self._step_counter += 1
#         # optional stride: update only every Nth call
#         if (int(self._step_counter.item()) % self.stride) != 0:
#             return self._S
#
#         for (c, d), S_src in attn_event_bqkl.items():
#             B, Q, K, L = S_src.shape
#             assert L == self.L and K == self.K
#             self._ensure_state((c, d), (B, Q, K), S_src.device)
#
#             mu = self._mu[(c, d)]  # [B, Q, K, L]
#             v  = self._sig(self.v_raw[f"{c}_{d}"])
#             alpha = torch.clamp(v * dt, 0.0, 1.0)
#
#             # upwind shift along lag axis: μ_ell <- (1-α) μ_ell + α μ_{ell+1}
#             mu_shift = torch.zeros_like(mu)
#             mu_shift[..., :-1] = mu[..., 1:]
#             mu = (1.0 - alpha) * mu + alpha * mu_shift
#
#             # source injection from current event attention (EMA-style)
#             if self.src_gain > 0:
#                 mu = (1.0 - self.src_gain) * mu + self.src_gain * torch.clamp(S_src, min=0)
#
#             mu = torch.clamp(mu, min=0)
#
#             # boundary flux at lag=1 goes into structural S (J = α * μ_old[...,0])
#             J = alpha * mu[..., 0]  # [B, Q, K]
#             S  = self._S[(c, d)]
#             S  = (1.0 - self.ema_kappa) * S + self.ema_kappa * J
#             # renormalize over K (avoid divide-by-zero)
#             S  = S / (S.sum(dim=-1, keepdim=True) + 1e-9)
#
#             # write back
#             self._mu[(c, d)] = mu
#             self._S[(c, d)]  = S
#
#         return self._S  # dict of [B,Q,K]
#
#
# class OptimalAttentionArbitration(nn.Module):
#     """
#     Closed-form optimal arbitration per type-pair and per query:
#       Q_k ∝ exp(-g_k/τ) * S_k^α * P_k^β * U_k^γ,  then normalize over K.
#     Inputs (dicts keyed by (c,d)):
#       - S_dict[(c,d)], P_dict[(c,d)]: [B, Q, K], row-stochastic over K
#       - U_dict[(c,d)] (optional):     [B, Q, K], prior (defaults to uniform)
#       - g_dict[(c,d)] (optional):     [B, Q, K], nonnegative per-edge cost (defaults to zeros)
#     Returns:
#       - Q_dict[(c,d)]:                 [B, Q, K], row-stochastic over K
#     """
#     def __init__(self,
#                  type_pairs,
#                  lambda_e: float = 1.0,
#                  lambda_s: float = 1.0,
#                  rho: float = 0.0,
#                  eps: float = 1e-12):
#         super().__init__()
#         self.type_pairs = list(type_pairs)
#         self.lambda_e = float(lambda_e)
#         self.lambda_s = float(lambda_s)
#         self.rho      = float(rho)
#         self.eps      = float(eps)
#
#         tau = self.lambda_e + self.lambda_s + self.rho
#         assert tau > 0.0, "At least one of lambda_e, lambda_s, rho must be > 0."
#         # store the normalized exponents (raw-independent)
#         self.alpha = self.lambda_e / tau
#         self.beta  = self.lambda_s / tau
#         self.gamma = self.rho      / tau
#         self.tau   = tau
#
#     def forward(self,
#                 S_dict: Dict[TypePair, torch.Tensor],
#                 P_dict: Dict[TypePair, torch.Tensor],
#                 U_dict: Dict[TypePair, torch.Tensor] = None,
#                 g_dict: Dict[TypePair, torch.Tensor] = None
#                 ) -> Dict[TypePair, torch.Tensor]:
#
#         Q_out: Dict[TypePair, torch.Tensor] = {}
#
#         for pair in self.type_pairs:
#             S = S_dict[pair]                 # [B,Q,K]
#             P = P_dict[pair]                 # [B,Q,K]
#             B, Q, K = S.shape
#
#             # Defaults for optional inputs, on the right device/dtype
#             if U_dict is None or pair not in U_dict:
#                 U = torch.ones_like(S) / K
#             else:
#                 U = U_dict[pair]
#             if g_dict is None or pair not in g_dict:
#                 g = torch.zeros_like(S)
#             else:
#                 g = g_dict[pair]
#
#             # Clamp to keep logs finite and preserve supports
#             eps = self.eps
#             S_ = torch.clamp(S, min=eps)
#             P_ = torch.clamp(P, min=eps)
#             U_ = torch.clamp(U, min=eps)
#
#             # log-domain multiplicative blend:
#             # log Q = -g/τ + α log S + β log P + γ log U, then softmax over K
#             logQ = -g / self.tau
#             if self.alpha != 0.0:
#                 logQ = logQ + self.alpha * torch.log(S_)
#             if self.beta != 0.0:
#                 logQ = logQ + self.beta  * torch.log(P_)
#             if self.gamma != 0.0:
#                 logQ = logQ + self.gamma * torch.log(U_)
#
#             Q_star = torch.softmax(logQ, dim=-1)  # [B,Q,K], row-stochastic over K
#             Q_out[pair] = Q_star
#
#         return Q_out
#
# # --- transport_ctmc.py -------------------------------------------------------
#
# class AttentionTransportCTMC(nn.Module):
#     """
#     CTMC transport on the lag axis with absorbing boundary at 0.
#     μ: [B,Q,K,L] row-stochastic over K per lag (kept approximately by convex mixing).
#     S: [B,Q,K] structural surrogate; mass at lag 0 after transport.
#     """
#     def __init__(self, type_pairs: List[TypePair], K: int, L: int,
#                  per_lag: bool = False,       # learn λ per lag or a single scalar
#                  ema_kappa: float = 0.5,
#                  src_gain: float = 0.2,
#                  stride: int = 2,
#                  device: str = "cuda"):
#         super().__init__()
#         self.type_pairs = list(type_pairs)
#         self.K, self.L = int(K), int(L)
#         self.ema_kappa = float(ema_kappa)
#         self.src_gain  = float(src_gain)
#         self.stride    = int(stride)
#
#         # Nonnegative rates λ (per pair, optionally per lag > 0)
#         if per_lag:
#             self.lam_raw = nn.ParameterDict({
#                 f"{c}_{d}": nn.Parameter(torch.zeros(L-1))  # softplus -> λ>=0
#                 for (c,d) in self.type_pairs
#             })
#         else:
#             self.lam_raw = nn.ParameterDict({
#                 f"{c}_{d}": nn.Parameter(torch.tensor(0.0))
#                 for (c,d) in self.type_pairs
#             })
#         self.per_lag = per_lag
#
#         # ephemeral state
#         self.register_buffer("_step_counter", torch.zeros((), dtype=torch.long))
#         self._mu: Dict[TypePair, torch.Tensor] = {}
#         self._S:  Dict[TypePair, torch.Tensor] = {}
#
#     @staticmethod
#     def _softplus(x): return F.softplus(x)  # ≥0
#
#     def reset_state(self):
#         self._mu.clear(); self._S.clear()
#         self._step_counter.zero_()
#
#     def _ensure_state(self, key: TypePair, B: int, Q: int, K: int, device: torch.device):
#         if key not in self._mu:
#             self._mu[key] = torch.zeros(B, Q, K, self.L, device=device)
#             self._S[key]  = torch.full((B, Q, K), 1.0 / K, device=device)
#
#     def _build_generator(self, pair: TypePair, device, dtype):
#         """Row-sum-zero Metzler generator G (LxL) with flow to smaller lags, 0 absorbing."""
#         L = self.L
#         G = torch.zeros(L, L, device=device, dtype=dtype)
#         lam_raw = self.lam_raw[f"{pair[0]}_{pair[1]}"]
#         if self.per_lag:
#             lam = self._softplus(lam_raw)            # [L-1]
#         else:
#             lam = self._softplus(lam_raw).expand(L-1)  # scalar -> [L-1]
#
#         # rows: destination; cols: source for matrix_exp semantics with row vector μ
#         for ell in range(1, L):
#             G[ell, ell]     = -lam[ell-1]   # outflow from lag ell
#             G[ell, ell-1]   =  lam[ell-1]   # inflow from ell-1? (NOTE: row-vector convention)
#         # Make 0 absorbing: row 0 is zeros
#         # Row sums are zero by construction; off-diagonal entries are ≥ 0
#         return G
#
#     def transport_step(self,
#                        attn_event_bqkl: Dict[TypePair, torch.Tensor],  # [B,Q,K,L]
#                        dt: float) -> Dict[TypePair, torch.Tensor]:
#         """
#         Evolve μ via μ_new = μ_prev @ exp(dt G); inject encoder source; update S by lag-0 mass.
#         Truncate gradients through stored state; grads flow to λ via matrix_exp this step.
#         """
#         self._step_counter += 1
#         do_update = (int(self._step_counter.item()) % max(self.stride,1) == 0)
#
#         out_S: Dict[TypePair, torch.Tensor] = {}
#         for pair, S_src in attn_event_bqkl.items():
#             B, Q, K, L = S_src.shape
#             assert L == self.L and K == self.K, "K/L mismatch"
#             dev, dty = S_src.device, S_src.dtype
#             self._ensure_state(pair, B, Q, K, dev)
#
#             mu_prev = self._mu[pair].detach()  # truncate-through-time
#             S_prev  = self._S[pair].detach()
#             S_src   = S_src.detach()           # keep encoder from chasing, safe default
#
#             if do_update and dt is not None:
#                 # exact row-stochastic transport via matrix exponential
#                 G  = self._build_generator(pair, dev, dty)      # [L,L]
#                 Kdt = torch.matrix_exp(G * float(dt))           # [L,L], row-stochastic
#
#                 # broadcast matmul over [B,Q,K, L] @ [L,L] -> [B,Q,K,L]
#                 mu_adv = torch.matmul(mu_prev, Kdt)
#
#                 # source injection (EMA with encoder lag histogram)
#                 if self.src_gain > 0.0:
#                     mu_new = (1.0 - self.src_gain) * mu_adv + self.src_gain * torch.clamp(S_src, min=0)
#                 else:
#                     mu_new = mu_adv
#                 mu_new = torch.clamp(mu_new, min=0)
#
#                 # boundary mass at lag 0 becomes structural surrogate (EMA)
#                 J = mu_new[..., 0]                              # [B,Q,K]
#                 S_new = (1.0 - self.ema_kappa) * S_prev + self.ema_kappa * J
#                 S_new = S_new / (S_new.sum(-1, keepdim=True) + 1e-9)
#
#                 # write back detached (truncate-through-time)
#                 self._mu[pair] = mu_new.detach()
#                 self._S[pair]  = S_new.detach()
#             # return differentiable S for this step (grads -> λ via matrix_exp path)
#             out_S[pair] = self._S[pair]
#
#         return out_S
#

# lib/attention_transport.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _matrix_exp_compat(A: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix exponential on all devices with autograd support.
    - If device supports torch.matrix_exp, use it.
    - If running on MPS (where matrix_exp is not available), compute on CPU and
      move the result back. Autograd will track the device transfers.
    Supports batched inputs [..., n, n].
    """
    dev_type = A.device.type
    # matrix_exp is available on CPU/CUDA. On MPS it's missing.
    if dev_type != "mps":
        return torch.matrix_exp(A)

    # Compute on CPU, preserving autograd
    A_cpu = A.to("cpu")
    K_cpu = torch.matrix_exp(A_cpu)
    return K_cpu.to(A.device)

# ------------ per-receiver-group normalization (edgewise) ---------------------

def group_normalize_edgewise(x_e: torch.Tensor,
                             group_id_e: torch.Tensor,
                             n_groups: int,
                             eps: float = 1e-9) -> torch.Tensor:
    """Normalize edge weights per receiver group (sum to 1 within each group)."""
    sums = x_e.new_zeros(n_groups)
    sums.index_add_(0, group_id_e, x_e)
    return x_e / (sums[group_id_e] + eps)

# ------------ CTMC transport on lag (edgewise) --------------------------------

class AttentionTransportCTMC_Edgewise(nn.Module):
    """
    Edgewise CTMC transport on lag with absorbing boundary at lag=0.

    State (per rollout):
      mu_eL : [E, L]  lag histogram per structural edge
      S_e   : [E]     boundary mass at lag 0 (structural surrogate)

    Learnables:
      lam_raw : λ_ℓ params (softplus -> nonnegative), either per-lag or shared.
    """
    def __init__(self, L: int, per_lag: bool = True, ema_kappa: float = 0.5, src_gain: float = 0.2):
        super().__init__()
        self.L = int(L)
        self.per_lag = bool(per_lag)
        self.ema_kappa = float(ema_kappa)
        self.src_gain = float(src_gain)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

        self.lam_raw = nn.Parameter(torch.zeros(self.L - 1) if per_lag else torch.tensor(0.0))

        # rollout state / metadata
        self.register_buffer("group_id_e", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("mu_eL",      torch.empty(0),                   persistent=False)
        self.register_buffer("S_e",        torch.empty(0),                   persistent=False)
        self.n_groups = None
        self.E_base   = None

    def set_groups(self, group_id_base: torch.Tensor, n_groups: int, E_base: int):
        """Set receiver-group ids for edges and group count."""
        self.group_id_e = group_id_base
        self.n_groups   = int(n_groups)
        self.E_base     = int(E_base)

    @torch.no_grad()
    def set_state_from(self, init_S_e: torch.Tensor = None):
        """Initialize μ and S; if init_S_e is None, start uniform per receiver."""
        assert self.group_id_e.numel() == self.E_base, "call set_groups() first"
        E = self.E_base
        device = self.group_id_e.device
        self.mu_eL = torch.zeros(E, self.L, device=device)
        if init_S_e is None:
            ones = torch.ones(E, device=device)
            deg  = torch.zeros(self.n_groups, device=device).index_add(0, self.group_id_e, ones)
            self.S_e = 1.0 / (deg[self.group_id_e] + 1e-9)
        else:
            self.S_e = group_normalize_edgewise(init_S_e.to(device), self.group_id_e, self.n_groups)

    def _generator(self, device, dtype):
        """Metzler generator G with flow ℓ→ℓ-1; lag 0 absorbing."""
        L = self.L
        G = torch.zeros(L, L, device=device, dtype=dtype)
        lam = self.softplus(self.lam_raw)
        if lam.ndim == 0:
            lam = lam.expand(L - 1)
        for ell in range(1, L):
            G[ell, ell]   = -lam[ell - 1]
            G[ell, ell-1] =  lam[ell - 1]
        return G

    def transport_step(self, attn_event_eL: torch.Tensor, dt: float, detach_src: bool = True) -> torch.Tensor:
        """
        attn_event_eL : [E, L]  encoder event attention binned by lag (edgewise).
        dt            : ODE step size.
        returns       : S_e_new [E] (group-normalized).
        """
        assert self.mu_eL.numel() and self.S_e.numel(), "call set_state_from() first"
        assert attn_event_eL.shape == self.mu_eL.shape, "shape mismatch"

        device, dtype = self.mu_eL.device, self.mu_eL.dtype
        G   = self._generator(device, dtype)          # [L,L]
        dt = float(dt)
        if abs(dt) < 1e-12:
            # No evolution; Kdt = I
            L = G.size(-1)
            Kdt = torch.eye(L, device=device, dtype=dtype)
        else:
            Kdt = _matrix_exp_compat(G * dt)  # [L,L]

        # Kdt = torch.matrix_exp(G * float(dt))         # [L,L]

        mu_adv = self.mu_eL @ Kdt                     # [E,L]

        S_src = attn_event_eL.detach() if detach_src else attn_event_eL
        mu_new = (1.0 - self.src_gain) * mu_adv + self.src_gain * torch.clamp(S_src, min=0)
        mu_new = torch.clamp(mu_new, min=0)

        J     = mu_new[:, 0]                          # [E]
        S_tmp = (1.0 - self.ema_kappa) * self.S_e + self.ema_kappa * J
        S_new = group_normalize_edgewise(S_tmp, self.group_id_e, self.n_groups)

        # TTT on state; gradients flow through Kdt → λ
        self.mu_eL = mu_new.detach()
        self.S_e   = S_new.detach()
        return S_new

# ------------ Optimal arbitration (edgewise, per receiver) --------------------

class OptimalAttentionArbitration_Edgewise(nn.Module):
    """
    Q_e ∝ exp(-g_e/τ) * S_e^α * P_e^β * U_e^γ (then group-normalize over incoming edges).
    """
    def __init__(self, lambda_e: float = 1.0, lambda_s: float = 1.0, rho: float = 0.0, eps: float = 1e-12):
        super().__init__()
        tau = float(lambda_e + lambda_s + rho)
        assert tau > 0.0
        self.alpha = float(lambda_e) / tau
        self.beta  = float(lambda_s) / tau
        self.gamma = float(rho)      / tau
        self.tau   = tau
        self.eps   = float(eps)
        self.group_id_e = None
        self.n_groups   = None

    def set_groups(self, group_id_base: torch.Tensor, n_groups: int):
        self.group_id_e = group_id_base
        self.n_groups   = int(n_groups)

    def forward(self,
                S_e: torch.Tensor,
                P_e: torch.Tensor,
                U_e: torch.Tensor = None,
                g_e: torch.Tensor = None) -> torch.Tensor:
        assert self.group_id_e is not None and self.n_groups is not None, "call set_groups() first"
        assert S_e.shape == P_e.shape, "S_e and P_e must be same shape [E]"
        E = S_e.numel()

        if U_e is None:
            ones = S_e.new_ones(E)
            deg  = ones.new_zeros(self.n_groups).index_add(0, self.group_id_e, ones)
            U_e  = 1.0 / (deg[self.group_id_e] + 1e-9)
        if g_e is None:
            g_e = torch.zeros_like(S_e)

        S_ = torch.clamp(S_e, min=self.eps)
        P_ = torch.clamp(P_e, min=self.eps)
        U_ = torch.clamp(U_e, min=self.eps)

        logQ = -g_e / self.tau
        if self.alpha: logQ = logQ + self.alpha * torch.log(S_)
        if self.beta:  logQ = logQ + self.beta  * torch.log(P_)
        if self.gamma: logQ = logQ + self.gamma * torch.log(U_)

        Q_raw = torch.exp(logQ)  # [E], positive
        return group_normalize_edgewise(Q_raw, self.group_id_e, self.n_groups)

# lib/attention_transport_neural.py
import torch, torch.nn as nn, torch.nn.functional as F

class AttentionTransportNeural_Edgewise(nn.Module):
    """
    Learn the edgewise lag-evolution μ_eL(t) with a mass-conserving neural generator.
    - Inputs (per edge): current μ_eL, encoder evidence S_src_eL, optional context φ_e
    - Output: S_e (boundary mass at lag 0) and next μ_eL

    Update: μ(t+dt) = μ(t) @ exp(G_e(μ,S_src,φ) * dt), with G_e Metzler (off-diag ≥0) and row-sum 0.
    We parameterize only the lower superdiagonal (flow ℓ->ℓ-1) via a NN and build G_e from it.
    """
    def __init__(self, L: int, hidden=64, ema_kappa: float = 0.5, src_gain: float = 0.2):
        super().__init__()
        self.L = int(L)
        self.ema_kappa = float(ema_kappa)
        self.src_gain = float(src_gain)

        in_dim = 2*L  # [μ_eL, S_src_eL]; you can concat extra context here
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, L-1)     # raw rates per lag
        )
        # state
        self.register_buffer("group_id_e", torch.empty(0, dtype=torch.long), persistent=False)
        self.n_groups = None
        self.E_base = None
        self.register_buffer("mu_eL", torch.empty(0), persistent=False)
        self.register_buffer("S_e", torch.empty(0), persistent=False)

    def set_groups(self, group_id_base: torch.Tensor, n_groups: int, E_base: int):
        self.group_id_e = group_id_base
        self.n_groups   = int(n_groups)
        self.E_base     = int(E_base)

    @torch.no_grad()
    def set_state_from(self, init_S_e: torch.Tensor = None):
        E = self.E_base
        device = self.group_id_e.device
        self.mu_eL = torch.zeros(E, self.L, device=device)
        if init_S_e is None:
            ones = torch.ones(E, device=device)
            deg  = torch.zeros(self.n_groups, device=device).index_add(0, self.group_id_e, ones)
            self.S_e = 1.0 / (deg[self.group_id_e] + 1e-9)
        else:
            self.S_e = self._group_norm(init_S_e)

    def _group_norm(self, x, eps=1e-9):
        sums = x.new_zeros(self.n_groups)
        sums.index_add_(0, self.group_id_e, x)
        return x / (sums[self.group_id_e].clamp_min(eps))

    def _build_generator(self, lam_e):  # lam_e: [E, L-1] nonneg
        E, L = lam_e.size(0), self.L
        # Build per-edge G_e as (batch of) lower-triangular band matrices
        # Using a minimal representation: only transitions ℓ->ℓ-1
        G = lam_e.new_zeros(E, L, L)
        idx = torch.arange(1, L, device=lam_e.device)
        G[:, idx, idx]     = -lam_e
        G[:, idx, idx-1]   =  lam_e
        return G

    def transport_step(self, attn_event_eL: torch.Tensor, dt: float) -> torch.Tensor:
        assert self.mu_eL.numel() and self.S_e.numel(), "call set_state_from() first"
        assert attn_event_eL.shape == self.mu_eL.shape

        # 1) infer per-edge rates from current state and evidence
        x = torch.cat([self.mu_eL, attn_event_eL], dim=-1)     # [E,L+L]
        lam_raw = self.net(x)                                   # [E,L-1]
        lam = F.softplus(lam_raw)                               # nonnegative

        # 2) advance distribution with matrix exponential of G_e
        G = self._build_generator(lam)                          # [E,L,L]

        dt = float(dt)
        if abs(dt) < 1e-12:
            # Kdt = I for all edges in the batch
            E, L = G.size(0), G.size(-1)
            I = torch.eye(L, device=G.device, dtype=G.dtype)
            Kdt = I.expand(E, L, L).contiguous()
        else:
            Kdt = _matrix_exp_compat(G * dt)  # [E,L,L]

        # Kdt = torch.matrix_exp(G * float(dt))                   # [E,L,L]
        mu_adv = torch.einsum('e lL, e Lm -> e lm', self.mu_eL, Kdt)  # [E,L]

        # 3) source injection from encoder evidence (EMA + source gain)
        mu_new = (1.0 - self.src_gain) * mu_adv + self.src_gain * attn_event_eL.clamp_min(0)
        mu_new = mu_new.clamp_min(0)

        # 4) boundary mass & group-normalize to S_e
        J = mu_new[:, 0]                                        # [E]
        S_tmp = (1.0 - self.ema_kappa) * self.S_e + self.ema_kappa * J
        S_new = self._group_norm(S_tmp)

        # persist state (stop gradients through state carry)
        self.mu_eL = mu_new.detach()
        self.S_e   = S_new.detach()
        return S_new

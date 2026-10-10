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

A third revision (CHANGES.md) added a training-only multi-horizon loss for extrapolation
(supervising several prefix cutoffs of the decode horizon, not just the full one) which measurably
narrowed, but did not close, the remaining gap on springs/charged-extrapolation. A fourth change
follows directly from that result:

4. **Second-order latent state** (GILODEFunc): the hidden state is split into two equal halves,
   h_i = [q_i, v_i], with dq_i/dt hard-wired to equal v_i (plus a small learned residual), and
   everything else -- the local term, the gated graph interaction -- feeding only into dv_i/dt.
   This targets exactly the regime the first three changes didn't touch: once extrapolation
   begins, no further correction ever arrives, and springs/charged are undamped, energy-
   conserving systems where vector-field error compounds without bound for the whole horizon.
   Removing one degree of freedom the network could get wrong (q's own dynamics, rather than
   letting it be learned freely and potentially inconsistent with v) is a standard technique for
   exactly this failure mode (Hamiltonian/Lagrangian-style neural ODEs). q/v are latent
   partitions, not literally physical position/velocity -- see the class docstring for why this
   is a deliberate simplification, not an oversight.

A fifth change follows from measuring *where* the remaining springs/charged-extrapolation gap
actually comes from. On springs-extrapolation the ranking is LG-ODE 2.02 / Edge-GNN 2.30 /
RNN-NRI 3.26 / GIL-ODE 4.62 / ODE-RNN 5.37 / Latent-ODE 5.60 -- and Edge-GNN, the *worst* model
on springs-interpolation, is second best here. What it shares with LG-ODE is that both encode
context once and then roll out a graph ODE that never corrects, so every gradient their vector
fields ever receive is long-rollout gradient. GIL-ODE's vector field, by contrast, spends
training doing short hops between corrections that keep rescuing it, then is judged on a long
uncorrected forecast it was barely trained for:

5. **Free-running supervision** (GILODEModel.forward + lib/baseline_gil_ode.py): during
   training, a second state is branched off the main path at a random point in the context
   window and rolled forward with corrections switched off, decoded at each subsequent
   observation and supervised against it. This is long unaided-rollout gradient drawn from the
   context observations, which are otherwise consumed only as corrections and never used as
   rollout targets. The main corrected path is untouched (the free-run state is a separate
   tensor), so unlike scheduled sampling this adds the missing training signal without degrading
   the primary objective.

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

# Ridge in the anchor gain (GILODEModel.anchor_gain), RELATIVE to the Gram matrix's scale so it is
# invariant to the decoder's magnitude. Large enough that the solve stays well conditioned when
# the decoder drifts (absolute 1e-6 and pinv both failed outright in practice), small enough that
# D(h^+) still matches the observation far inside the 0.01 observation std these datasets use.
ANCHOR_RTOL = 1e-5



class GILODEFunc(nn.Module):
    '''
    Second-order latent state (see module docstring, point 4): h_i = [q_i, v_i], split into two
    equal halves of the hidden state, with a hard-wired integrability constraint:
        dq_i/dt = v_i                       -- q's derivative IS v, exactly (no residual term)
        dv_i/dt = f_self(h_i) + g_i(t) * f_interaction(h_i, m_i)  -- everything else lives here
    q/v are latent partitions, not literally physical position/velocity (the encoder/decoder are
    unchanged plain linear layers over the full state, so nothing forces q or v to align with any
    specific raw feature) -- the point of the split is the hard integrability constraint itself,
    which removes one degree of freedom a free-form vector field could get wrong over a long,
    uncorrected rollout, not a claim that q "is" position for any given dataset. g_i(t) is the
    per-node, state-dependent gate and m_i is the learned sum+mean combination of relation-expert
    messages, both unchanged from the previous revision (this file's earlier docstring section);
    only where their output feeds into (v's derivative only, not q's) has changed.
    '''

    def __init__(self, hidden_dim, mlp_width=None, adapter_width=120, adapter_gate_width=32):
        super(GILODEFunc, self).__init__()
        # MLP width is decoupled from the ODE state dimension: a rollout state that is too wide
        # drifts over a long uncorrected forecast (measured: springs-extrap 3.391 at state 152 vs
        # 2.389 at state 80), but the vector field still needs capacity. So extra parameters go
        # into the networks, not into the state being integrated.
        w = mlp_width if mlp_width is not None else hidden_dim
        self.q_dim = hidden_dim // 2
        self.v_dim = hidden_dim - self.q_dim

        # dv/dt's local term (was f_local, over the full state, output now v-sized).
        self.f_self = utils.create_net(hidden_dim, self.v_dim, n_layers=1, n_units=w, nonlinear=nn.Tanh)
        # NOTE: an eps_q residual on dq/dt was tried and removed. It was intended as a *small*
        # correction on top of the hard "dq/dt = v" skeleton, but nothing constrained its
        # magnitude, and a trained checkpoint showed ||eps_q(h)|| at 2.37x ||v|| -- i.e. the
        # network routed around the constraint entirely, leaving the position half free-form
        # again while still paying the constraint's cost (f_self/f_interaction halved to v_dim).
        # dq/dt = v exactly, so the constraint actually binds. See CHANGES.md.
        # Relation-expert messages, including the pairwise difference (h_j - h_i).
        self.phi_pos = utils.create_net(hidden_dim * 3, hidden_dim, n_layers=2, n_units=w, nonlinear=nn.Tanh)
        self.phi_neg = utils.create_net(hidden_dim * 3, hidden_dim, n_layers=2, n_units=w, nonlinear=nn.Tanh)
        # Antisymmetric force channel: f_ij = chi(h_i, h_j) - chi(h_j, h_i), which satisfies
        # f_ij = -f_ji identically by construction -- Newton's third law, which nothing in the
        # relation-expert messages above enforces (they compute i<-j and j<-i independently).
        # This is exactly right for springs, whose force is Hooke's law F_ij = -k(x_i - x_j):
        # antisymmetric AND linear in the pairwise difference. It is offered as an ADDITIONAL
        # channel rather than replacing the free-form messages, so datasets whose dynamics are
        # not a clean pairwise force law (IEEE39's swing equation with damping and control) can
        # down-weight it via psi instead of being forced into the wrong prior.
        self.chi = utils.create_net(hidden_dim * 2, hidden_dim, n_layers=1, n_units=w, nonlinear=nn.Tanh)
        # Learned combination of sum- and mean-aggregated messages, the antisymmetric channel,
        # and log(1+degree).
        self.psi = utils.create_net(hidden_dim * 3 + 1, hidden_dim, n_layers=1, n_units=w, nonlinear=nn.Tanh)
        # f_interaction(h_i, m_i) -- the gated term added to dv/dt only.
        self.f_interaction = utils.create_net(hidden_dim * 2, self.v_dim, n_layers=1, n_units=w, nonlinear=nn.Tanh)
        # State-dependent gate: sigmoid(a_theta([h_i, m_i, log(1+deg_i)]) + b0), b0 chosen so the
        # gate starts near 0.1 -- same warm-start rationale as the scalar alpha it replaces
        # (a hard-zero start left no gradient incentive to grow under long horizons).
        self.gate_net = utils.create_net(hidden_dim * 2 + 1, 1, n_layers=1, n_units=w, nonlinear=nn.Tanh)

        utils.init_network_weights(self.f_self)
        utils.init_network_weights(self.phi_pos)
        utils.init_network_weights(self.phi_neg)
        utils.init_network_weights(self.chi)
        utils.init_network_weights(self.psi)
        utils.init_network_weights(self.f_interaction)
        utils.init_network_weights(self.gate_net)
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(-2.1972)  # logit(0.1)

        # ---- Horizon-gated residual expert (stagewise boosting of the vector field) ----
        # dv/dt gains + a(h, tau) * R(h), where tau is time since this node's last observation.
        # Both the residual's output layer and the gate's output layer start at exactly zero, so
        # a freshly-adapted model reproduces its base checkpoint bit-for-bit and can only depart
        # from it by learning to. The base is frozen while this trains, so the five winning cells
        # cannot regress by construction -- the residual has to earn any change it makes.
        #
        # Gated on tau specifically because that is what separates the tasks: interpolation keeps
        # tau small (observations keep arriving, corrections keep landing), so the residual stays
        # near-dormant and the existing behaviour is preserved; extrapolation lets tau grow
        # without bound, which is exactly where accumulated drift and unmodelled events (springs'
        # wall collisions) actually hurt. It routes into dv only, never dq -- dq/dt = v is a hard
        # constraint that measurably helps, and a residual on dq would dissolve it the same way
        # eps_q did above.
        # n_layers=0 gives a single hidden layer (create_net emits Linear, act, Linear); n_layers=1
        # would give two and put the adapter at ~33K, over budget.
        self.residual = utils.create_net(hidden_dim, self.v_dim, n_layers=0, n_units=adapter_width,
                                         nonlinear=nn.Tanh)
        self.residual_gate = utils.create_net(hidden_dim + 1, 1, n_layers=0, n_units=adapter_gate_width,
                                              nonlinear=nn.Tanh)
        utils.init_network_weights(self.residual)
        utils.init_network_weights(self.residual_gate)
        with torch.no_grad():
            # Zeroing the residual's output layer is what makes the adapter an exact no-op, so the
            # adapted model starts identical to its base checkpoint regardless of the gate.
            self.residual[-1].weight.zero_()
            self.residual[-1].bias.zero_()
            # The gate is additionally warm-started near 0.1 rather than sigmoid(0)=0.5. Once the
            # residual has any nonzero output, a gate sitting at 0.5 would be half-active at every
            # tau -- including the small-tau interpolation regime whose wins this is supposed to
            # protect. Starting dormant means the residual must earn activation, and the tau input
            # makes it cheapest to earn where tau is large. Same rationale, and same value, as
            # gate_net's own warm start above.
            self.residual_gate[-1].weight.zero_()
            self.residual_gate[-1].bias.fill_(-2.1972)  # logit(0.1)

        # ---- State-dependent stability guard ----
        # Radial damping on the decoder-null-space component n = h - h P, applied ONLY once ||n||
        # exceeds a calibrated radius R:
        #
        #     dv <- dv - gamma * relu(||n|| - R) * n_v / (||n|| + eps)
        #
        # Below R the relu is exactly zero, so this is bit-for-bit inactive in the regime the five
        # winning cells operate in -- it cannot regress them. Above R it pulls the null component
        # back radially, which is the measured failure mode on springs extrapolation: the
        # unconstrained subspace drifts (3.86x across the forecast), the state grows, the learned
        # dynamics stiffen, and fixed-step rk4 blows up around epoch 4 (CHANGES.md Part 26.7).
        #
        # Applied to dv rather than the full dh: a damping term on dq would dissolve the hard
        # dq/dt = v constraint the same way eps_q did. Damping the velocity along the null
        # direction still bounds the position component, since q can only grow through v -- this is
        # a drag force, not a positional spring.
        #
        # gamma starts near zero (softplus(-5) ~ 0.0067) so the guard begins almost off and must
        # earn its magnitude, and is a single scalar trained uniformly -- no per-dataset value.
        self.log_gamma = nn.Parameter(torch.tensor(0.0))
        # R is calibrated from normal training states (mean + 3 std of ||n||, ~the 99.7th
        # percentile under normality) by the same procedure on every dataset, then frozen. Held as
        # a buffer so it travels with the checkpoint. Negative means "not yet calibrated" -> guard
        # inactive, which is also what makes the calibration pass itself unguarded.
        self.register_buffer('guard_R', torch.tensor(-1.0))

        # Ablation switches (paper Table 2). Empty in every reported model.
        self.ablate = set()

        self.S = None
        self.c = None
        self._tau_base = None
        self._t0 = None
        self._P_row = None

    def set_graph(self, S, c):
        self.S = S  # [B, N, N]
        self.c = c  # [B, N, N]

    def set_null_projector(self, P_row):
        '''Row-space projector P = K W, so h - h P is the decoder-null component the guard acts
        on. Set per forward from the current decoder, since W is still training.'''
        self._P_row = P_row

    def set_tau(self, tau_base, t0):
        '''
        Time since each node's last observation, as of the start of the current integration
        segment. odeint only hands forward() an absolute t, so tau at time t inside the segment is
        tau_base + (t - t0) -- tau advances with integration time and is reset by the caller
        whenever an observation lands.
        '''
        self._tau_base = tau_base  # [B, N]
        self._t0 = t0

    def forward(self, t, h):
        B, N, H = h.shape
        v = h[..., self.q_dim:]

        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)
        edge_in = torch.cat([h_i, h_j, h_j - h_i], dim=-1)

        is_pos = (self.c > 0).unsqueeze(-1)
        msg = torch.where(is_pos, self.phi_pos(edge_in), self.phi_neg(edge_in))  # [B, N, N, H]

        # Antisymmetric force channel: f_ij = chi(h_i,h_j) - chi(h_j,h_i) satisfies f_ij = -f_ji
        # exactly. chi is evaluated once and its transpose reused, so this costs one extra MLP
        # call over the edge tensor, not two.
        chi_ij = self.chi(torch.cat([h_i, h_j], dim=-1))  # [B, N, N, H]
        f_anti = chi_ij - chi_ij.transpose(1, 2)

        S_ = self.S.unsqueeze(-1)
        deg = self.S.sum(dim=2, keepdim=True)  # [B, N, 1]
        m_sum = (S_ * msg).sum(dim=2)  # [B, N, H]
        m_mean = m_sum / deg.clamp(min=1.0)
        m_anti = (S_ * f_anti).sum(dim=2)  # [B, N, H] -- additive, as a net force should be
        if 'no_antisym' in self.ablate:
            m_anti = torch.zeros_like(m_anti)
        log_deg = torch.log1p(deg)  # [B, N, 1]
        m_tilde = self.psi(torch.cat([m_sum, m_mean, m_anti, log_deg], dim=-1))  # [B, N, H]

        gate_in = torch.cat([h, m_tilde, log_deg], dim=-1)
        g = torch.sigmoid(self.gate_net(gate_in))  # [B, N, 1]
        interaction = self.f_interaction(torch.cat([h, m_tilde], dim=-1))  # [B, N, v_dim]

        dq = v  # [B, N, q_dim] -- hard, binding: q's derivative IS v, no escape hatch
        dv = self.f_self(h) + g * interaction  # [B, N, v_dim]

        # Horizon-gated residual expert. Zero at initialization (both output layers are zeroed),
        # so this is exactly a no-op until trained -- an adapted model starts identical to its
        # base checkpoint. log1p(tau) rather than raw tau so the gate sees the same input scale
        # across datasets whose time units differ, and is not saturated by a long forecast.
        if self._tau_base is not None and 'no_residual' not in self.ablate:
            tau = (self._tau_base + (t - self._t0)).unsqueeze(-1)  # [B, N, 1]
            a = torch.sigmoid(self.residual_gate(torch.cat([h, torch.log1p(tau.clamp(min=0))], dim=-1)))
            dv = dv + a * self.residual(h)

        # Stability guard. Exactly zero wherever ||n|| <= R, so this is inert in the normal
        # operating regime and only engages on abnormal drift. See __init__.
        if self._P_row is not None and float(self.guard_R) > 0.0 and 'no_guard' not in self.ablate:
            n = h - h @ self._P_row                             # [B, N, H] null-space component
            n_norm = n.norm(dim=-1, keepdim=True)               # [B, N, 1]
            excess = torch.relu(n_norm - self.guard_R)          # 0 below the radius
            gamma = F.softplus(self.log_gamma)
            dv = dv - gamma * excess * (n[..., self.q_dim:] / (n_norm + 1e-8))

        return torch.cat([dq, dv], dim=-1)


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

    def __init__(self, input_dim, hidden_dim, num_atoms, device, mlp_width=None, ode_substeps=1, ode_tol=None, ode_max_depth=6,
                 use_forecast_adapter=False, forecast_adapter_rank=4, use_smoother=False,
                 smoother_mode='full', ablate=(), use_reversion=False, n_particles=1):
        super(GILODEModel, self).__init__()
        # Probabilistic GIL-ODE (paper/probabilistic_design.md, CHANGES.md Part 30). With
        # n_particles = K > 1 the latent state is an ensemble of K particles following the SDE
        #   dh = f(h) dt + diag(sigma) dW
        # (split-step: the deterministic RK4 drift over a segment, then one Euler-Maruyama
        # diffusion increment sqrt(dt) * sigma * eps). Every particle is anchored and lifted
        # exactly as before, so with noiseless measurements observed agents collapse onto y in the
        # observable subspace while the null-space spread survives. The point forecast is the
        # ensemble mean; the samples are kept in self.last_samples for the CRPS term.
        self.n_particles = n_particles
        if n_particles > 1:
            # softplus(-3.0) ~ 0.049 per sqrt(time unit): small, and learned.
            self.log_diffusion = nn.Parameter(torch.full((hidden_dim,), -3.0))
        # Ablations (paper Table 2), all off in the reported model:
        #   no_lift      unobserved agents get no correction (delta_U = 0): per-agent update only
        #   overwrite    replace subspace anchoring by h_obs <- E(y) (the pre-anchoring rule)
        #   no_gate      beta = 1 for every unobserved agent
        #   complete     lift and propagate on S = 1 - I instead of the physical graph
        #   no_residual / no_guard / no_antisym   drop that term from the vector field
        #   no_null_prior is handled in the training wrapper
        self.ablate = set(ablate)
        # Evaluation only: decode the PRIOR state h^- at every grid time (before that time's
        # observation is applied), so a target that coincides with an observation is predicted
        # without seeing it. Interpolation targets ARE the conditioning observations (LG-ODE's
        # protocol), and the anchored posterior reproduces them by construction.
        self.decode_prior = False
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.device = device
        self.ode_substeps = ode_substeps
        self.ode_tol = ode_tol          # None -> fixed step; float -> error-controlled rk4
        self.ode_max_depth = ode_max_depth
        self._refine_count = 0

        # No encoder projection. Anchoring used to overwrite the whole latent state with
        # E_theta(y), which pushed 4-5 observed features into all hidden_dim dimensions and
        # discarded everything the state had accumulated; the correction is now confined to the
        # decoder's row space instead (see anchor_gain / forward), so no y -> h map is needed.
        # Forecast-transition adapter (rank r, shared across nodes), applied once at the last
        # observation to turn the filtering state into a forecasting state. fa_U's weight is
        # zeroed, so the whole thing is a bit-exact no-op at initialization and any checkpoint
        # that predates it loads with these parameters still zero -- existing results are
        # unaffected. Disabled entirely unless use_forecast_adapter is set.
        self.use_forecast_adapter = use_forecast_adapter
        self.use_smoother = use_smoother
        # 'full': inverse-distance blend wherever a future observation exists.
        # 'coldstart': backward state ONLY before a node's first observation, forward everywhere else.
        self.smoother_mode = smoother_mode
        self.fa_V = nn.Linear(hidden_dim, forecast_adapter_rank, bias=True)
        self.fa_U = nn.Linear(forecast_adapter_rank, hidden_dim, bias=False)
        utils.init_network_weights(self.fa_V)
        with torch.no_grad():
            self.fa_U.weight.zero_()

        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.ode_func = GILODEFunc(hidden_dim, mlp_width=mlp_width)
        # Stability-guard calibration state (see forward / finalize_guard_radius).
        self.calibrating = False
        self._calib_n = 0
        self._calib_sum = 0.0
        self._calib_sumsq = 0.0
        self.lifting = GraphLifting(hidden_dim)
        self.gate = GateNet()

        # ---- Mean-reverting forecast head (CHANGES.md Part 29) ----
        # Between observations the anchored state is a point estimate whose uncertainty grows
        # with the time tau_i since agent i was last observed. Under chaotic coupling (charged
        # particles' close encounters) the MSE-optimal forecast then relaxes from the rolled-out
        # trajectory toward a conditional mean, as a Kalman forecast's would, instead of
        # committing to one sharp path. Measured before building this: a per-step blend toward the
        # global mean, fitted on validation only, cut charged-extrap test MSE 6.24 -> 5.54 and was
        # ~0 on springs (run_logs/eval/shrink_analysis.py).
        #
        #   y_hat_i(t) = (1 - s_i) D(h_i(t)) + s_i mu_i
        #   s_i  = sigmoid(kappa([h_i(t), log(1 + tau_i)]))           reversion gate
        #   mu_i = M [h_i(t_i^last), mean_j h_j(t_j^last)] + m         graph-conditional mean
        #
        # mu is read from the states at each agent's last observation (the information the
        # forecast is conditioned on) together with the graph average, so it is per trajectory,
        # not a global constant. The gate's output layer is zero with bias logit(0.01), so the head
        # starts as an (almost) exact no-op and engages only where the loss pays for it.
        self.use_reversion = use_reversion
        if use_reversion:
            self.rev_gate = utils.create_net(hidden_dim + 1, 1, n_layers=0, n_units=32, nonlinear=nn.Tanh)
            self.rev_mean = nn.Linear(2 * hidden_dim, input_dim)
            utils.init_network_weights(self.rev_gate)
            utils.init_network_weights(self.rev_mean)
            with torch.no_grad():
                self.rev_gate[-1].weight.zero_()
                self.rev_gate[-1].bias.fill_(-4.595)   # logit(0.01)
        utils.init_network_weights(self.decoder)
        self.ode_func.ablate = self.ablate
        if 'overwrite' in self.ablate:
            self.encoder_proj = nn.Linear(input_dim, hidden_dim)
            utils.init_network_weights(self.encoder_proj)

    def _innovation(self, y, h, mask_f, K):
        '''Latent correction for observed agents (zero elsewhere): the anchored K e, or, under the
        'overwrite' ablation, E(y) - h, which resets the whole state to an encoding of y.'''
        if 'overwrite' in self.ablate:
            return mask_f.unsqueeze(-1) * (self.encoder_proj(y) - h)
        return mask_f.unsqueeze(-1) * ((y - self.decoder(h)) @ K.transpose(0, 1))

    def _event_update(self, h, tso, S, c, r, mask_f):
        N = h.shape[1]
        delta = self.lifting(h, S, c, r, mask_f)
        mask_col = mask_f.unsqueeze(-1)
        if 'no_lift' in self.ablate:
            return h + mask_col * delta
        obs_density = mask_f.mean(dim=1, keepdim=True).expand(-1, N)
        beta = self.gate(tso, obs_density, delta.norm(dim=-1))
        if 'no_gate' in self.ablate:
            beta = torch.ones_like(beta)
        return h + mask_col * delta + (1 - mask_col) * beta * delta

    def _integrate(self, h, t0, t1, depth=0):
        '''
        Error-controlled rk4 by step doubling (Richardson): take one full step, take two half
        steps, and compare. If they agree to within ode_tol the full step is already resolving the
        dynamics, so its result is kept -- unchanged from the plain fixed-step behaviour. If they
        disagree the interval is recursively bisected, each half getting the same test.

        This is one global rule with one tolerance, applied identically to every dataset and both
        tasks. Where the fixed step was already accurate (the five cells that never blew up) it
        returns exactly what it returned before and costs only the error check; where it was not
        (springs extrapolation, where a single step per grid segment diverges at epoch 4 and an 8x
        finer fixed step trains monotonically) it refines automatically, and only as far as
        needed.

        The two half steps are taken under no_grad: they exist only to estimate the error, and the
        returned value is always the graph-carrying `full` (accepted) or the graph-carrying result
        of the recursion (refined), so this adds no autograd graph of its own when it accepts.
        '''
        full = odeint(self.ode_func, h, torch.stack([t0, t1]), method='rk4')[-1]
        if depth >= self.ode_max_depth:
            return full

        # The interval can become too short to bisect: once (t1 - t0) approaches the dtype's
        # resolution, 0.5*(t0+t1) rounds to t0 or t1 and odeint rejects the non-increasing span.
        # The main loop only guarantees segments > 1e-8, and six levels of bisection divides that
        # by 64, so this is reachable -- IEEE39's denser grid hits it on the first batch. Accept
        # the full step there; a step that short is already resolving the dynamics.
        t_mid = 0.5 * (t0 + t1)
        lo, hi = (t0, t1) if t0 < t1 else (t1, t0)   # backward pass integrates with t0 > t1
        if not (lo < t_mid and t_mid < hi):
            return full

        with torch.no_grad():
            h_mid = odeint(self.ode_func, h.detach(), torch.stack([t0, t_mid]), method='rk4')[-1]
            h_two = odeint(self.ode_func, h_mid, torch.stack([t_mid, t1]), method='rk4')[-1]
            err = (full.detach() - h_two).abs().max().item()

        if err <= self.ode_tol:
            return full

        self._refine_count += 1
        t_mid = 0.5 * (t0 + t1)
        left = self._integrate(h, t0, t_mid, depth + 1)
        return self._integrate(left, t_mid, t1, depth + 1)

    def finalize_guard_radius(self, n_std=3.0):
        '''
        Freeze the stability guard's radius at mean + n_std * std of the null-component norms seen
        during calibration -- approximately the 99.7th percentile under normality, i.e. a radius
        normal training states essentially never reach. Identical procedure and identical n_std on
        every dataset; the radius differs between datasets only because their state scales do.
        Returns the radius, or None if nothing was collected.
        '''
        self.calibrating = False
        if self._calib_n == 0:
            return None
        mean = self._calib_sum / self._calib_n
        var = max(self._calib_sumsq / self._calib_n - mean ** 2, 0.0)
        radius = mean + n_std * (var ** 0.5)
        self.ode_func.guard_R.fill_(radius)
        return radius

    def anchor_gain(self):
        '''
        Minimum-norm gain that makes the decoded state reproduce an observation exactly.

        The decoder is linear, D(h) = h W^T + b, so for an observation-space innovation
        e = y - D(h^-) the correction K e with

            K = W^T (W W^T + eps I)^-1

        gives D(h^- + K e) = D(h^-) + W K e ~= D(h^-) + e = y. K's columns span W's row space,
        so the update moves the state only within the subspace the decoder can actually see and
        leaves its orthogonal complement -- the latent memory carried between observations --
        untouched. That is the whole point: the previous rule (h^+ = E_theta(y)) reconstructed
        the observation equally well but erased that memory at every single observation event,
        which is exactly what ODE-RNN avoids by combining y with the previous hidden state.

        Recomputed per forward because W is still training, and NOT detached -- gradients flow
        through the gain on purpose. The intuition that this is dangerous (a near-singular W
        yields an enormous K, and the anchor forces D(h^+) = y however large K is, so nothing
        obviously penalizes it) turned out to be exactly backwards when measured. Detaching was
        tried and left the decoder free to drift: condition number 1.3 -> 2.9 on charged and
        1.6 -> 5.0 on IEEE39 over 50 epochs, with ||K|| growing 2.20 -> 3.47, and interpolation
        degraded accordingly (charged 0.2195 -> 0.2658). With gradients connected the decoder
        stays well conditioned (1.3 -> 1.6, ||K|| flat at ~2.1), because the loss can now feel
        that an inflated gain produces large damaging corrections. The gradient path is a
        stabilizer, not an attractor (CHANGES.md Part 26).

        Solved as a Tikhonov system with a SCALE-RELATIVE ridge rather than via pinv or a fixed
        absolute eps, both of which have now failed in practice: an absolute eps=1e-6 solve raised
        "input matrix is singular" on springs extrapolation at epoch 29, and pinv later raised
        "svd failed to converge" on the same cell at epoch 14 once free-running was active. Adding
        a positive multiple of I to the PSD Gram matrix makes it positive definite by
        construction, so this solve cannot fail, and scaling the ridge by the Gram's mean diagonal
        makes it invariant to the decoder's overall magnitude. If W does approach rank deficiency
        the gain stays bounded and the anchor degrades smoothly from exact to approximate, instead
        of the run dying.
        '''
        W = self.decoder.weight                                       # [D, H]
        G = W @ W.transpose(0, 1)                                     # [D, D]
        scale = G.diagonal(dim1=-2, dim2=-1).mean().clamp(min=1e-12)
        eye = torch.eye(G.shape[0], device=W.device, dtype=W.dtype)
        return torch.linalg.solve(G + ANCHOR_RTOL * scale * eye, W).transpose(0, 1)  # [H, D]

    def _diffuse(self, h, dt):
        '''Euler-Maruyama diffusion increment over a segment of length |dt| (no-op if K = 1).'''
        if self.n_particles == 1:
            return h
        sigma = F.softplus(self.log_diffusion)
        return h + sigma * torch.sqrt(torch.abs(dt)) * torch.randn_like(h)

    def forward(self, dense, obs_mask, grid_times, decoder_time_steps, S, c, free_run_from=None,
                free_run_span=None, free_run_detach=False):
        '''Ensemble wrapper: with K particles, fold them into the batch dimension, run the
        filter once, and return the ensemble-mean prediction (samples in self.last_samples).'''
        K = self.n_particles
        if K == 1:
            return self._forward_single(dense, obs_mask, grid_times, decoder_time_steps, S, c,
                                        free_run_from, free_run_span, free_run_detach)
        B = dense.shape[0]
        rep_ = lambda x: x.repeat_interleave(K, dim=0)
        pred, free_run, null_penalty = self._forward_single(
            rep_(dense), rep_(obs_mask), grid_times, decoder_time_steps, rep_(S), rep_(c),
            free_run_from, free_run_span, free_run_detach)
        samples = pred.view(B, K, *pred.shape[1:])                  # [B, K, N, T_Q, D]
        self.last_samples = samples
        return samples.mean(1), free_run, null_penalty

    def _forward_single(self, dense, obs_mask, grid_times, decoder_time_steps, S, c, free_run_from=None,
                        free_run_span=None, free_run_detach=False):
        '''
        dense: [B, N, T, D], obs_mask: [B, N, T] bool, grid_times: [T] (union of every node's
        observation times and the decoder's query times, ascending), S/c: [B, N, N].
        Returns pred_x [B, N, T_Q, D] at decoder_time_steps.

        free_run_from (training only, see module docstring point 5): if given, an index into
        grid_times from which a SECOND, parallel state is rolled forward with corrections
        switched off, decoded at every subsequent observation, and returned for supervision
        against those observations. This is the only place the vector field gets long unaided
        rollout gradient from the context window -- along the main path, corrections keep
        rescuing it every few steps, so it is otherwise trained almost entirely on short hops
        while being judged on a long uncorrected forecast. The main corrected path is left
        completely untouched by this (the free-run state is a separate tensor), so the primary
        loss is unaffected -- unlike scheduled sampling, which degrades the main path itself.

        free_run_span sets how long any one unaided stretch is allowed to run, in time units.
        Left None, a single branch is taken at free_run_from and runs to the end of the grid --
        the right shape for extrapolation, where the model really does run uncorrected from the
        forecast boundary onward. Given a span, the free state is re-branched off the main path
        whenever it has run unaided for that long, so the window is tiled by many short
        rollouts instead of one long one. That is the right shape for interpolation, where the
        longest unaided stretch the model ever faces is the gap between two observations; a
        full-length branch there trains for a rollout the task never asks for, and measurably
        costs assimilation quality (CHANGES.md Part 25).
        '''
        B, N, T, D = dense.shape
        if 'complete' in self.ablate:
            S = (1 - torch.eye(N, device=S.device)).expand(B, -1, -1)
        self.ode_func.set_graph(S, c)
        K = self.anchor_gain()  # [H, D] -- W is fixed within a forward pass, so compute it once
        P_row = K @ self.decoder.weight                            # [H, H] row-space projector
        self.ode_func.set_null_projector(P_row)                    # for the stability guard
        self._diag_corr_max = 0.0

        # Forecast-transition adapter: applied ONCE, at the last observation, to convert the
        # filtering state into a forecasting state. See fa_U / fa_V in __init__. Only the index is
        # needed here; the adapter itself is applied inside the loop.
        last_obs_idx = -1
        if self.use_forecast_adapter:
            nz = obs_mask.any(dim=1).any(dim=0).nonzero()
            if nz.numel() > 0:
                last_obs_idx = int(nz.max().item())
        self._refine_count = 0

        h = torch.zeros(B, N, self.hidden_dim, device=self.device)
        time_since_obs = torch.zeros(B, N, device=self.device)
        outputs = torch.zeros(B, N, T, self.hidden_dim, device=self.device)

        h_free = None
        free_branch_t = None  # grid time at which the current unaided stretch started
        free_preds, free_targets, free_masks = [], [], []

        prev_t = grid_times[0]
        for i in range(T):
            cur_t = grid_times[i]
            if i > 0 and (cur_t - prev_t).abs() > 1e-8:
                # tau as of this segment's start; the vector field advances it with integration
                # time internally (GILODEFunc.set_tau).
                self.ode_func.set_tau(time_since_obs, prev_t)

                if self.ode_tol is not None:
                    # Error-controlled rk4: keeps the plain one-step result wherever it already
                    # agrees with two half steps, refines recursively where it does not.
                    h = self._integrate(h, prev_t, cur_t)
                elif self.ode_substeps > 1:
                    # Fixed finer step -- diagnostic path used to establish that the
                    # springs-extrap blowup was integration error (CHANGES.md Part 26).
                    t_span = torch.linspace(float(prev_t), float(cur_t), self.ode_substeps + 1,
                                            device=grid_times.device, dtype=grid_times.dtype)
                    h = odeint(self.ode_func, h, t_span, method='rk4')[-1]
                else:
                    h = odeint(self.ode_func, h, torch.stack([prev_t, cur_t]), method='rk4')[-1]

                h = self._diffuse(h, cur_t - prev_t)
                time_since_obs = time_since_obs + (cur_t - prev_t)
                if h_free is not None:
                    h_free = (self._integrate(h_free, prev_t, cur_t) if self.ode_tol is not None
                              else odeint(self.ode_func, h_free,
                                          torch.stack([prev_t, cur_t]), method='rk4')[-1])
                    h_free = self._diffuse(h_free, cur_t - prev_t)

            if free_run_from is not None and i >= free_run_from:
                # Branch off the main path, and, when a span is set, re-branch once this stretch
                # has run unaided for its full length. h here is post-integration but
                # pre-correction, so a branch always starts from a state the corrections have not
                # yet touched at this step.
                #
                # free_run_detach severs the branch from the main path's history, so the
                # free-running loss cannot backpropagate through the assimilation machinery
                # (GraphLifting / GateNet / encoder_proj) that produced h. The branch state
                # becomes "wherever assimilation put us", treated as a constant. The auxiliary
                # loss exists to train rollout dynamics, and this is what confines it to that
                # job -- paired with restricting its parameter set in the caller. Without the
                # detach, gradients would still reach those modules through the state's history
                # even if their parameters were excluded from the update.
                branch = h.detach() if free_run_detach else h
                if h_free is None:
                    h_free, free_branch_t = branch, cur_t
                elif free_run_span is not None and (cur_t - free_branch_t) > free_run_span:
                    h_free, free_branch_t = branch, cur_t

            h_prior = h
            mask_i = obs_mask[:, :, i]  # [B, N] bool
            if mask_i.any():
                mask_f = mask_i.float()
                y_i = dense[:, :, i, :]

                # Supervise only where the free state has actually run unaided for some time --
                # at the instant of a (re-)branch it is just a copy of the main path, so it
                # carries no rollout signal.
                if h_free is not None and (cur_t - free_branch_t).abs() > 1e-8:
                    free_preds.append(self.decoder(h_free))
                    free_targets.append(y_i)
                    free_masks.append(mask_f)

                # Observation-subspace anchor: e is the innovation measured where it is actually
                # observable (decoder output space), and K e is the smallest latent correction
                # that explains it. Was `encoder_proj(y_i) - h`, which drove h to E_theta(y_i)
                # outright and so overwrote all hidden_dim dimensions from input_dim observed
                # features at every event. See anchor_gain.
                r = self._innovation(y_i, h, mask_f, K)               # [B, N, H]
                with torch.no_grad():
                    # Anchor-correction magnitude. The gain K = W^T (W W^T + ridge)^-1 scales like
                    # 1/sigma_min(W), which the decoder's CONDITION number cannot detect -- cond is
                    # scale-invariant, so a uniformly shrinking W keeps cond flat while ||K|| and
                    # hence every correction blows up. Tracked explicitly for that reason.
                    _c = r.norm(dim=-1).max().item()
                    if _c > self._diag_corr_max:
                        self._diag_corr_max = _c
                # Observed nodes get their exact correction unconditionally (delta_i == r_i,
                # hard-anchored in GraphLifting); the gate only weighs *inferred* corrections
                # for unobserved nodes -- see module docstring, point 1.
                h = self._event_update(h, time_since_obs, S, c, r, mask_f)
                time_since_obs = torch.where(mask_i, torch.zeros_like(time_since_obs), time_since_obs)

            # Forecast transition. GIL-ODE's latent state is trained to support repeated
            # observation corrections; at the last observation it must abruptly become a state
            # that runs autonomously for the whole forecast. LG-ODE never makes that transition --
            # it builds one context-derived initial state specifically for autonomous rollout,
            # which is the most plausible account of why it leads this cell. This adapter learns
            # that conversion:
            #
            #     h_forecast = h_filtered + P_null U tanh(V h_filtered)
            #
            # The update is projected into the decoder's NULL space, so D(h_forecast) = D(h_filtered):
            # it cannot alter the reconstructed position/velocity at the final observation, only
            # reorganize the hidden information the rollout will use. U is zero-initialized, so it
            # is a bit-exact no-op until trained -- checkpoints without these parameters (every
            # existing run, including the five winning cells) behave exactly as before.
            if self.use_forecast_adapter and i == last_obs_idx:
                d = self.fa_U(torch.tanh(self.fa_V(h)))            # [B, N, H]
                h = h + d - d @ P_row                              # d projected onto null(W)

            outputs[:, :, i] = h_prior if self.decode_prior else h
            prev_t = cur_t

        # Zero-mean Gaussian prior on the unobservable latent state. Anchoring constrains only the
        # input_dim directions the decoder can see; the remaining hidden_dim - input_dim (76 of 80
        # here) are touched by no observation and bounded by nothing in the loss, and they duly
        # inflate -- measured on springs extrapolation, ||h_null|| grows 3.86x across the forecast
        # against the observable part's 1.96x, ending 2.3x larger than the component the decoder
        # can actually see, and growing monotonically even inside the observed window.
        #
        # The overwrite rule this replaced held that subspace at zero implicitly, by resetting the
        # whole state at every observation; dropping it left an improper flat prior in its place.
        # This restores a proper one: weak enough to let the subspace carry memory between
        # observations, which was the entire point of subspace anchoring, but enough to stop it
        # growing without bound over a long uncorrected rollout. P = K W is the row-space
        # projector, so h - h P is exactly the part no observation can reach.
        P = K @ self.decoder.weight                      # [H, H], symmetric idempotent
        h_null = outputs - outputs @ P
        null_penalty = h_null.pow(2).sum(dim=-1).mean()

        # Calibrate the stability guard's radius from normal training states. Reuses h_null, which
        # the prior above already computes, so this costs one norm and two reductions. Runs only
        # while self.calibrating is set (the first epoch, before any instability appears -- the
        # springs-extrap blowup is consistently around epoch 4), then the radius is frozen.
        if self.calibrating:
            with torch.no_grad():
                nn_norm = h_null.norm(dim=-1).reshape(-1)
                self._calib_n += nn_norm.numel()
                self._calib_sum += nn_norm.sum().item()
                self._calib_sumsq += nn_norm.pow(2).sum().item()

        # Instability diagnostics, recorded per forward. These separate the three candidate causes
        # of the springs-extrap blowup: an exploding anchor gain (row space), a growing latent
        # state / vector field (energy injection), or pure integration error.
        with torch.no_grad():
            f_final = self.ode_func(grid_times[-1], outputs[:, :, -1])
            self.last_diag = {
                'K_norm': K.norm().item(),
                'corr_max': self._diag_corr_max,
                'h_final': outputs[:, :, -1].norm(dim=-1).mean().item(),
                'h_max': outputs.norm(dim=-1).max().item(),
                'f_norm': f_final.norm(dim=-1).mean().item(),
                'refines': self._refine_count,
            }

        query_idx = torch.searchsorted(grid_times, decoder_time_steps)
        h_at_query = outputs[:, :, query_idx]

        # ---- Bidirectional smoothing (interpolation) ----
        # The forward pass is a causal FILTER: its state at time t has seen only observations up to t.
        # Measured on springs interpolation, 99.9% of its error comes from targets BEFORE a node's
        # first observation (1.5% of targets, MSE 4.3e-2 there vs 1.3e-6 between observations) -- a
        # cold start, where the filter has seen nothing about that node yet. Latent-ODE (backward
        # encoder) and RNN-NRI (bidirectional GRU) both carry later observations back in time, which
        # is exactly where they beat GIL. This adds the smoother counterpart: the same dynamics,
        # anchoring, lifting and gate run backwards in time, fused with the forward state.
        #
        # Fusion weight on the backward state is tau_f / (tau_f + tau_b), with tau_f the time since
        # the previous observation and tau_b the time until the next, per node:
        #   before a node's first observation (tau_f = inf)  -> backward only;
        #   after its last observation (tau_b = inf)          -> forward only;
        #   on an observation (tau_f = 0)                     -> forward (anchored exactly);
        #   between observations                               -> inverse-distance blend.
        # Parameter-free. In extrapolation every target lies after every observation, so tau_b = inf
        # everywhere, the backward pass is skipped, and the output is bit-identical to the filter.
        if self.use_smoother:
            Q = decoder_time_steps.shape[0]
            OT = grid_times.view(1, 1, 1, T).expand(B, N, Q, T)
            OM = obs_mask.unsqueeze(2).expand(B, N, Q, T)
            TQ = decoder_time_steps.view(1, 1, Q, 1)
            prev_obs = torch.where(OM & (OT <= TQ + 1e-9), OT, torch.full_like(OT, -float('inf'))).amax(-1)
            next_obs = torch.where(OM & (OT >= TQ - 1e-9), OT, torch.full_like(OT, float('inf'))).amin(-1)
            tau_f = decoder_time_steps.view(1, 1, Q) - prev_obs     # [B, N, Q], inf if none before
            tau_b = next_obs - decoder_time_steps.view(1, 1, Q)     # [B, N, Q], inf if none after
            fin_f, fin_b = torch.isfinite(tau_f), torch.isfinite(tau_b)

            if fin_b.any():
                h_b = torch.zeros(B, N, self.hidden_dim, device=self.device)
                tso_b = torch.zeros(B, N, device=self.device)
                outputs_b = torch.zeros(B, N, T, self.hidden_dim, device=self.device)
                nxt_t = grid_times[-1]
                for i in reversed(range(T)):
                    cur_t = grid_times[i]
                    if i < T - 1 and (nxt_t - cur_t).abs() > 1e-8:
                        self.ode_func.set_tau(tso_b, nxt_t)
                        if self.ode_tol is not None:
                            h_b = self._integrate(h_b, nxt_t, cur_t)
                        else:
                            h_b = odeint(self.ode_func, h_b, torch.stack([nxt_t, cur_t]), method='rk4')[-1]
                        h_b = self._diffuse(h_b, nxt_t - cur_t)
                        tso_b = tso_b + (nxt_t - cur_t)
                    h_b_prior = h_b
                    mask_i = obs_mask[:, :, i]
                    if mask_i.any():
                        mask_f = mask_i.float()
                        r = self._innovation(dense[:, :, i, :], h_b, mask_f, K)
                        h_b = self._event_update(h_b, tso_b, S, c, r, mask_f)
                        tso_b = torch.where(mask_i, torch.zeros_like(tso_b), tso_b)
                    outputs_b[:, :, i] = h_b_prior if self.decode_prior else h_b
                    nxt_t = cur_t

                both = fin_f & fin_b
                cold = ~fin_f & fin_b                                  # before this node's first observation
                if self.smoother_mode == 'coldstart':
                    w_b = torch.where(cold, torch.ones_like(tau_f), torch.zeros_like(tau_f))
                else:
                    w_b = torch.where(both, tau_f / (tau_f + tau_b).clamp(min=1e-12),
                                      torch.where(cold, torch.ones_like(tau_f), torch.zeros_like(tau_f)))
                w_b = torch.nan_to_num(w_b, nan=0.0).unsqueeze(-1)   # [B, N, Q, 1]
                h_at_query = (1 - w_b) * h_at_query + w_b * outputs_b[:, :, query_idx]

                # The backward states carry the same unconstrained null-space subspace as the forward
                # ones, so they get the same prior.
                hb_null = outputs_b - outputs_b @ P
                null_penalty = 0.5 * (null_penalty + hb_null.pow(2).sum(dim=-1).mean())

        pred = self.decoder(h_at_query)

        if self.use_reversion:
            # Each agent's last observation index on the grid, its state there, and the time from
            # it to every query. Agents never observed in the window fall back to the window start.
            Q = decoder_time_steps.shape[0]
            idx = torch.arange(T, device=obs_mask.device).view(1, 1, T).expand(B, N, T)
            last = torch.where(obs_mask, idx, torch.zeros_like(idx)).amax(-1)            # [B, N]
            h_last = torch.gather(outputs, 2, last.view(B, N, 1, 1).expand(B, N, 1, self.hidden_dim)).squeeze(2)
            mu = self.rev_mean(torch.cat([h_last, h_last.mean(1, keepdim=True).expand(-1, N, -1)], -1))  # [B, N, D]
            tau_q = (decoder_time_steps.view(1, 1, Q) - grid_times[last].unsqueeze(-1)).clamp(min=0)    # [B, N, Q]
            s_rev = torch.sigmoid(self.rev_gate(torch.cat([h_at_query, torch.log1p(tau_q).unsqueeze(-1)], -1)))
            pred = (1 - s_rev) * pred + s_rev * mu.unsqueeze(2)
            self.last_diag_rev = float(s_rev.mean())

        if free_preds:
            free_run = (torch.stack(free_preds, dim=2),    # [B, N, K, D]
                        torch.stack(free_targets, dim=2),  # [B, N, K, D]
                        torch.stack(free_masks, dim=2))    # [B, N, K]
        else:
            free_run = None
        return pred, free_run, null_penalty

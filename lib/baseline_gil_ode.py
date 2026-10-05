'''
Wraps lib/gil_ode.py's GILODEModel with the same likelihood/MSE computation every other model
in this project uses (reusing VAE_Baseline's get_gaussian_likelihood/get_mse, which don't
depend on the KL/z0_prior machinery -- same reuse pattern as lib/baseline_odernn.py). GIL-ODE is
deterministic (no z0 posterior), so there is no KL term.

Multi-horizon training curriculum (extrapolation only, CHANGES.md Part 23): once forecasting
starts, GIL-ODE's vector field runs unaided for the whole horizon with no further corrections,
and until now the loss only ever supervised the full decode horizon at once, giving every decode
point equal, undifferentiated weight. This adds loss terms at several intermediate prefix
cutoffs of the decode horizon (all computed from the single existing forward pass -- GIL-ODE
already produces predictions at every decoder time step in one shot, so this is free in terms of
forward compute, just extra loss bookkeeping), which gives early-horizon accuracy more effective
gradient weight (an early point contributes to every cutoff's term; only the last point
contributes to just the final one) without changing the model or requiring a new forward pass
per horizon. Interpolation has no meaningful "horizon away from a fixed boundary" (decode points
are scattered through the observed window, not a monotonic future sequence), so this only
applies in extrap mode.
'''
import torch

from lib.base_models import VAE_Baseline
from lib.gil_ode import GILODEModel
from lib.gil_dataset import prepare_gil_batch
from lib.likelihood_eval import gaussian_log_likelihood

HORIZON_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
# Relative to the primary likelihood term; both use the same 1/(2 sigma^2) scale, so this is a
# true relative weight. Was 1.0, inherited from Part 23 and never re-validated -- the term was
# silently inactive from Part 25 until the free_run_from bug was found, so no run since Part 24
# exercised it. Reinstated at 1.0 it contributed ~5x the reconstruction term at initialization,
# rising to ~7.5x by epoch 11 (rollout error falls more slowly than reconstruction error), and
# springs extrapolation diverged twice: train MSE reached 0.00068 then blew up 45x to 0.030 and
# the weights went non-finite by epoch 15. The decoder stayed well conditioned throughout
# (cond 1.28 -> 1.24), so this was divergence, not the rank collapse first suspected.
# 0.2 puts the auxiliary term at roughly parity with the primary objective at initialization
# instead of dominating it. One value for every dataset (CHANGES.md Part 26).
FREE_RUN_WEIGHT = 0.2
# Weight on the unobservable-subspace prior (lib/gil_ode.py). Deliberately weak: the subspace is
# meant to carry memory between observations, and the prior only has to stop it growing without
# bound, so this is calibrated to sit at a few percent of the reconstruction term rather than to
# drive the null component to zero. One value for every dataset and both tasks.
NULL_PRIOR_WEIGHT = 1.0


def _pooled_masked_mse(pred_flat, truth_slice, mask_slice):
    '''
    Batch-pooled (not per-trajectory-then-averaged) masked MSE for one horizon cutoff, with a
    clamped denominator -- unlike VAE_Baseline.get_mse (which this deliberately doesn't reuse),
    a short prefix cutoff can leave some node with zero observed target points in that window,
    and get_mse's per-trajectory normalization (lib/likelihood_eval.py:
    compute_masked_likelihood) divides by that count directly with no floor, which is exactly
    what produced NaN when first tried. Pooling across the whole batch before dividing sidesteps
    this without touching the shared function every other model relies on.
    '''
    mask_rep = mask_slice.unsqueeze(0)
    se = (pred_flat - truth_slice.unsqueeze(0)) ** 2
    denom = mask_rep.sum().clamp(min=1.0)
    return (se * mask_rep).sum() / denom


def _pooled_masked_gaussian_ll(pred_flat, truth_slice, mask_slice, obsrv_std):
    '''Batch-pooled masked Gaussian log-likelihood for one horizon cutoff -- see
    _pooled_masked_mse for why this doesn't reuse VAE_Baseline.get_gaussian_likelihood.'''
    mask_rep = mask_slice.unsqueeze(0)
    log_prob = gaussian_log_likelihood(pred_flat, truth_slice.unsqueeze(0), obsrv_std=obsrv_std)
    denom = mask_rep.sum().clamp(min=1.0)
    return (log_prob * mask_rep).sum() / denom


class GILODEBaseline(VAE_Baseline):

    def __init__(self, input_dim, hidden_dim, num_atoms, dataset, obsrv_std, device, mode="interp",
                 mlp_width=None, ode_substeps=1, ode_tol=None, use_forecast_adapter=False,
                 free_run=False, free_run_weight=FREE_RUN_WEIGHT, use_smoother=False,
                 smoother_mode='full'):
        super(GILODEBaseline, self).__init__(
            input_dim=input_dim, latent_dim=hidden_dim, z0_prior=None, device=device, obsrv_std=obsrv_std)
        self.num_atoms = num_atoms
        self.dataset = dataset
        self.mode = mode
        # Free-running supervision, re-enabled for testing under ERROR-CONTROLLED integration.
        # Every previous divergence was measured under fixed-step rk4, which is now known to have
        # been failing on its own (CHANGES.md Part 26), so those rejections are confounded.
        self.free_run = free_run
        self.free_run_weight = free_run_weight
        self.core = GILODEModel(input_dim, hidden_dim, num_atoms, device, mlp_width=mlp_width,
                                ode_substeps=ode_substeps, ode_tol=ode_tol,
                                use_forecast_adapter=use_forecast_adapter,
                                use_smoother=use_smoother, smoother_mode=smoother_mode)

    def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                            n_traj_samples=1, kl_coef=1.):
        dense, mask, grid_times, S, c = prepare_gil_batch(
            batch_dict_encoder, batch_dict_decoder, batch_dict_graph, self.num_atoms, self.dataset, self.device)

        decoder_time_steps = batch_dict_decoder["time_steps"]

        # Free-running supervision (training only): branch an uncorrected rollout off the main
        # path inside the context window, so the vector field gets unaided rollout gradient
        # instead of only short between-correction hops (lib/gil_ode.py, point 5).
        #
        # FULLY DISABLED. It is off for interpolation because every way of enabling it there
        # traded one dataset against another, and off for extrapolation because every way of
        # enabling it there diverged. Both bodies of evidence are recorded below, since the code
        # is retained and someone will be tempted to switch it back on.
        #
        # INTERPOLATION -- four variants, each bought springs-interp at charged's and/or IEEE39's
        # expense (CHANGES.md Part 26):
        #   full-horizon branch: sp 0.068 win, ie 0.891 -> 1.341 loss
        #   gap-sized tiled branches: sp 0.0656 win, ch 0.2613 -> 0.2995, ie -> 1.216 both loss
        #   global PCGrad gate: provably inert -- the two gradients are ALIGNED (mean cos +0.88,
        #     never negative in 40 steps), so it reproduced ungated free-running exactly
        #   gradient isolation to the vector field: sp 0.0654 win, ch 0.2839 loss, ie ~1.36,
        #     its worst result of the three configs
        # Per-layer diagnostics explain why no routing rule fixes this: the conflict is in the
        # assimilation modules on springs (20-30% of steps) but in the *vector field* on charged
        # (phi_pos/phi_neg, 40-45%) and essentially absent on IEEE39 -- there is no single set of
        # parameters to protect. The springs-interp gap was closed by the anchoring change
        # instead, which is uniform by construction.
        #
        # EXTRAPOLATION -- every variant diverged on springs: weight 1.0 (epoch 15), weight
        # 0.2 (epoch 15, identical epoch at 5x the weight, which ruled magnitude out), a
        # horizon-matched branch point (epoch 15) and the random branch point Parts 23/24 actually
        # validated (epoch 9). Decoder conditioning stayed healthy throughout every one of them
        # (cond 1.28 -> 1.35), so these were divergences, not the rank collapse first suspected.
        #
        # The interaction is with anchoring itself, which is what changed underneath this
        # mechanism. Parts 23/24 ran free-running against the old full-overwrite rule, which reset
        # the entire latent state at every observation; subspace anchoring deliberately preserves
        # the hidden_dim - input_dim directions no observation constrains, and the free-run branch
        # then rolls precisely those unconstrained, drifting directions forward uncorrected with
        # gradients flowing through the whole rollout.
        #
        # Note this term was ALREADY contributing exactly zero from Part 25 onward because of the
        # branch-point bug, so no result since Part 24 -- including Part 25's 5/6 table -- was
        # produced with it active. Disabling it changes nothing about those numbers; it only makes
        # the intent explicit. Re-enabling needs the drift interaction solved first, not a new
        # weight.
        free_run_from = None
        if self.free_run and self.training and self.mode == "extrap":
            T_grid = grid_times.shape[0]
            obs_steps = mask.any(dim=1).any(dim=0).nonzero()
            if T_grid > 4 and obs_steps.numel() > 1:
                # Branch at a RANDOM point in the context, resampled per batch -- the scheme
                # Parts 23/24 validated (springs-extrap 4.620 -> 3.391). Two prior attempts to
                # make this horizon-matched instead both failed (CHANGES.md Part 26):
                #   Part 25 measured the horizon back from grid_times[-1], which in extrapolation
                #     is the last decoder QUERY, beyond every observation -- so the branch landed
                #     past the final observation, nothing followed it to supervise, and the term
                #     contributed exactly zero for the whole Part 25 era without anyone noticing.
                #   Measuring back from the last observation instead put the branch a full
                #     forecast horizon (~40 steps) earlier, backpropagating through that entire
                #     uncorrected rollout; springs-extrap then diverged at epoch 15 at both
                #     weight 1.0 and 0.2 (train MSE bottoming at 0.0015 then climbing to 0.107,
                #     decoder conditioning healthy at 1.28->1.35 the whole time, so divergence
                #     rather than rank collapse). Identical failure epoch at a 5x weight change
                #     is what identified the span, not the magnitude, as the cause.
                # So horizon-matching has never once trained successfully, while the random point
                # has. Sampling a fresh branch point per batch also varies the rollout length
                # seen across training rather than fixing it at one span.
                last_obs_idx = int(obs_steps.max().item())
                if last_obs_idx > 2:
                    free_run_from = int(torch.randint(1, last_obs_idx - 1, (1,)).item())

        pred, free_run, null_penalty = self.core(dense, mask, grid_times, decoder_time_steps, S, c,
                                                 free_run_from=free_run_from)  # [B, N, T_Q, D]

        B, N, T_Q, D = pred.shape
        pred_flat = pred.reshape(B * N, T_Q, D).unsqueeze(0)  # [1, M, T_Q, D]

        truth = batch_dict_decoder["data"]       # [M, T_Q, D]
        target_mask = batch_dict_decoder["mask"]  # [M, T_Q, D]

        if self.mode == "extrap" and T_Q > 1:
            cutoffs = sorted(set(max(1, round(T_Q * f)) for f in HORIZON_FRACTIONS))
            liks = [_pooled_masked_gaussian_ll(pred_flat[:, :, :H], truth[:, :H], target_mask[:, :H], self.obsrv_std)
                    for H in cutoffs]
            mses = [_pooled_masked_mse(pred_flat[:, :, :H], truth[:, :H], target_mask[:, :H])
                    for H in cutoffs]
            rec_likelihood = torch.stack(liks).mean().unsqueeze(0)
            mse_val = torch.stack(mses).mean()
        else:
            rec_likelihood = self.get_gaussian_likelihood(truth, pred_flat, None, mask=target_mask)
            mse_val = self.get_mse(truth, pred_flat, mask=target_mask)

        loss = -torch.mean(rec_likelihood)

        # The encoder/decoder consistency term that used to sit here is gone, subsumed by the
        # anchoring change. It existed because anchoring set an observed node's state to
        # encoder_proj(y) while reconstruction read out decoder(h), so every decode target
        # coinciding with an observation carried decoder(encoder_proj(y)) - y as avoidable error,
        # and nothing in the end-to-end loss pushed on that composition. Observation-subspace
        # anchoring makes D(h^+) = y hold exactly by construction (lib/gil_ode.py anchor_gain),
        # so there is no residual left for a penalty to close, and encoder_proj no longer exists.
        if free_run is not None:
            fr_pred, fr_target, fr_mask = free_run
            fr_mask = fr_mask.unsqueeze(-1)  # [B, N, K, 1] -> broadcasts over features
            fr_se = ((fr_pred - fr_target) ** 2) * fr_mask
            fr_mse = fr_se.sum() / (fr_mask.sum() * fr_pred.shape[-1]).clamp(min=1.0)
            # Same 1/(2*sigma^2) scaling the primary likelihood uses, so the two terms are
            # commensurate and FREE_RUN_WEIGHT is a true relative weight, not a scale fudge.
            loss = loss + self.free_run_weight * fr_mse / (2 * self.obsrv_std ** 2).squeeze()

        # Prior on the unobservable latent subspace (lib/gil_ode.py). Applied in training only and
        # identically in both modes and all three datasets -- it is a property of the state space,
        # not of a task or a dataset.
        if self.training:
            loss = loss + NULL_PRIOR_WEIGHT * null_penalty

        return {
            "loss": loss,
            "null_penalty": null_penalty.data.item(),
            "likelihood": torch.mean(rec_likelihood).data.item(),
            "mse": torch.mean(mse_val).data.item(),
            "kl_first_p": 0.0,  # no single scalar to track anymore -- gate is now per-node/state-dependent
            "std_first_p": 0.0,
        }

'''
GIL-ODE (Graph Innovation-Lifting ODE): the proposed architecture (see lib/gil_ode.py's
docstring for the full design). Reuses the same corrected dataloaders as
run_models_corrected.py (CorrectedParseData / IEEE39ParseData) for a fair, apples-to-apples
comparison against Corrected LG-ODE and every baseline -- same data, splits, normalization, and
observation masks. Unlike the other baselines, the graph tensor IS used here (to build the
per-dataset support/relation matrices S/c, see lib/gil_dataset.py).
'''
import os
import sys
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from lib.baseline_gil_ode import GILODEBaseline
from lib.utils import compute_loss_all_batches

parser = argparse.ArgumentParser('GIL-ODE')
parser.add_argument('--n-balls', type=int, default=5)
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--save', type=str, default='experiments_gilode/')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('-r', '--random-seed', type=int, default=1991)
parser.add_argument('--data', type=str, default='spring', help="spring,charged,ieee39")
parser.add_argument('--hidden-dim', type=int, default=80)
parser.add_argument('--mlp-width', type=int, default=92, help='MLP width, decoupled from the ODE state dim (see lib/gil_ode.py)')
parser.add_argument('--extrap', type=str, default="False")
parser.add_argument('--sample-percent-train', type=float, default=0.6)
parser.add_argument('--sample-percent-test', type=float, default=0.6)
parser.add_argument('--val-fraction', type=float, default=None, help='Override CorrectedParseData.VAL_FRACTION (e.g. for fixed-size subsets to hit an exact split)')
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default="AdamW")
parser.add_argument('--clip', type=float, default=10)
parser.add_argument('--cutting_edge', type=bool, default=True)
parser.add_argument('--extrap_num', type=int, default=40)
parser.add_argument('--alias', type=str, default="run")
parser.add_argument('--horizon-loss', type=str, default='prefix', choices=['prefix', 'full'],
                    help='extrap training loss: mean over 20..100%% horizon prefixes, or the full horizon only')
parser.add_argument('--reversion', action='store_true',
                    help='mean-reverting forecast head (lib/gil_ode.py, CHANGES.md Part 29)')
parser.add_argument('--particles', type=int, default=1,
                    help='K > 1: probabilistic GIL-ODE, ensemble of K stochastic particles (CHANGES.md Part 30)')
parser.add_argument('--ablate', type=str, default='',
                    help='comma-separated ablations (lib/gil_ode.py): no_lift, overwrite, no_gate, complete, '
                         'no_residual, no_guard, no_antisym, no_null_prior')
parser.add_argument('--dataset-dir', type=str, default=None)
parser.add_argument('--ode-tol', type=float, default=None, help='Enable error-controlled rk4 (step doubling): keep the one-step result where it agrees with two half steps to within this tolerance, else recursively bisect. One global value for every dataset.')
parser.add_argument('--ode-substeps', type=int, default=1, help='Fixed rk4 substeps per grid segment; >1 shrinks the step size proportionally')
parser.add_argument('--smoother', action='store_true',
                    help='Bidirectional smoothing: run the GIL correction/lifting machinery backwards in '
                         'time too and fuse with the forward filter by inverse-distance weighting. A '
                         'no-op in extrapolation (no target has a future observation).')
parser.add_argument('--smoother-mode', type=str, default='coldstart', choices=['coldstart', 'full'],
                    help="coldstart: backward state only before a node's first observation (forward "
                         "elsewhere). full: inverse-distance blend wherever a future observation exists.")
parser.add_argument('--free-run', action='store_true',
                    help='Re-enable free-running supervision (extrapolation only). Every prior '
                         'rejection of it was measured under the broken fixed-step integrator.')
parser.add_argument('--free-run-weight', type=float, default=1.0,
                    help='Weight on the free-running term; 1.0 is the Parts 23/24 validated value.')
parser.add_argument('--forecast-adapter-from', type=str, default=None,
                    help='Train ONLY the forecast-transition adapter (fa_U/fa_V, ~660 params) on '
                         'top of a loaded checkpoint, with every other parameter frozen. The '
                         'adapter is a bit-exact no-op at init (fa_U zeroed) and applies once at '
                         'the last observation, projected into the decoder null space, so it '
                         'cannot change what the model reconstructs there. Validation-selected '
                         'against the loaded checkpoint, which is kept as the fallback.')
parser.add_argument('--refine-from', type=str, default=None,
                    help='Forecast-phase vector-field refinement. Loads a checkpoint, freezes '
                         'everything except the continuous vector field (encoder/anchoring, '
                         'lifting, gate, decoder all frozen), and trains only that on the ordinary '
                         'extrapolation objective, with a proximal penalty keeping it near the '
                         'loaded solution. Validation-selected; the starting checkpoint is '
                         'evaluated first so refinement can never be selected unless it improves.')
parser.add_argument('--prox-weight', type=float, default=0.0,
                    help='Weight on ||theta_F - theta_F_checkpoint||^2 for --refine-from.')
parser.add_argument('--lr-plateau', action='store_true',
                    help='Validation-triggered LR reduction WITH ROLLBACK, replacing the cosine '
                         'schedule. On 3 consecutive epochs without validation improvement: '
                         'restore the best-validation weights, rebuild the optimizer (clearing '
                         'momentum), and multiply LR by 0.2. At most 3 reductions '
                         '(5e-4 -> 1e-4 -> 2e-5 -> 4e-6); stop after 8 epochs without improvement '
                         'at the final LR. Checkpoint selection stays purely validation-based.')
parser.add_argument('--adapt-from', type=str, default=None,
                    help='Path to a base checkpoint. Loads it, FREEZES every base parameter, and '
                         'trains only the horizon-gated residual adapter (lib/gil_ode.py). The '
                         'adapter is a bit-exact no-op at init, so training starts from exactly '
                         'the base model and can only depart from it by learning to.')

args = parser.parse_args()

if args.data == "spring":
    args.dataset = 'data/spring'
    args.suffix = '_springs5'
    args.total_ode_step = 60
elif args.data == "charged":
    args.dataset = 'data/charged'
    args.suffix = '_charged5'
    args.total_ode_step = 60
elif args.data == "ieee39":
    args.dataset = 'data/processed/ieee39_gen'
    args.suffix = '_ieee39gen'
    args.total_ode_step = 60
    args.n_balls = 10

if args.dataset_dir is not None:
    args.dataset = args.dataset_dir

if torch.cuda.is_available():
    print("Using GPU" + "-" * 80)
    device = torch.device("cuda:0")
else:
    print("Using CPU" + "-" * 80)
    device = torch.device("cpu")

if args.extrap == "True":
    print("Running extrap mode" + "-" * 80)
    args.mode = "extrap"
else:
    print("Running interp mode" + "-" * 80)
    args.mode = "interp"


if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    utils.makedirs(args.save)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    print("Loading dataset: " + args.dataset)
    if args.data == "ieee39":
        from lib.ieee39_dataLoader import IEEE39ParseData
        dataloader = IEEE39ParseData(args.dataset, mode=args.mode, args=args)
    else:
        from lib.corrected_dataLoader import CorrectedParseData
        dataloader = CorrectedParseData(args.dataset, suffix=args.suffix, mode=args.mode, args=args)

    train_encoder, train_decoder, train_graph, train_batch = dataloader.load_data(
        sample_percent=args.sample_percent_train, batch_size=args.batch_size, data_type="train")
    val_encoder, val_decoder, val_graph, val_batch = dataloader.load_data(
        sample_percent=args.sample_percent_test, batch_size=args.batch_size, data_type="val")
    test_encoder, test_decoder, test_graph, test_batch = dataloader.load_data(
        sample_percent=args.sample_percent_test, batch_size=args.batch_size, data_type="test")

    input_dim = dataloader.feature

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    obsrv_std = torch.Tensor([0.01]).to(device)
    model = GILODEBaseline(input_dim=input_dim, hidden_dim=args.hidden_dim, num_atoms=args.n_balls,
                           dataset=args.data, obsrv_std=obsrv_std, device=device, mode=args.mode,
                           mlp_width=args.mlp_width, ode_substeps=args.ode_substeps, ode_tol=args.ode_tol,
                           use_forecast_adapter=(args.forecast_adapter_from is not None),
                           free_run=args.free_run, free_run_weight=args.free_run_weight,
                           use_smoother=args.smoother, smoother_mode=args.smoother_mode,
                           ablate=tuple(a for a in args.ablate.split(',') if a),
                           horizon_loss=args.horizon_loss, use_reversion=args.reversion,
                           n_particles=args.particles).to(device)

    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)

    if args.forecast_adapter_from is not None:
        # Checkpoints predate the adapter, so fa_U/fa_V are absent -- they keep their zero init,
        # which is exactly the intended starting point (bit-exact no-op). Anything else missing
        # would be a real architecture mismatch.
        state = torch.load(args.forecast_adapter_from, map_location=device, weights_only=False)['state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        fa_prefixes = ('core.fa_U.', 'core.fa_V.')
        unrelated = [k for k in missing if not k.startswith(fa_prefixes)]
        if unrelated or unexpected:
            raise RuntimeError(f"architecture mismatch -- missing: {unrelated}; unexpected: {list(unexpected)}")
        n_frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith(fa_prefixes):
                p.requires_grad_(False); n_frozen += p.numel()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Forecast-adapter training from {args.forecast_adapter_from}")
        print(f"  frozen   : {n_frozen:,}")
        print(f"  trainable: {n_train:,}")

    prox_ref = None
    if args.refine_from is not None:
        state = torch.load(args.refine_from, map_location=device, weights_only=False)['state_dict']
        model.load_state_dict(state, strict=True)
        # Only the continuous vector field stays trainable. Everything responsible for
        # assimilation -- anchoring, innovation lifting, the correction gate -- and the decoder are
        # frozen at their checkpoint values, so the five winning cells' behaviour is untouched and
        # only the component responsible for autonomous rollout is updated.
        n_frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith('core.ode_func.'):
                p.requires_grad_(False)
                n_frozen += p.numel()
        # Proximal anchor: keep the refined field near the solution that already works.
        prox_ref = {n: p.detach().clone() for n, p in model.named_parameters()
                    if n.startswith('core.ode_func.') and p.requires_grad}
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Refining vector field from {args.refine_from}")
        print(f"  frozen  : {n_frozen:,}")
        print(f"  trainable (vector field): {n_train:,}")
        print(f"  prox weight: {args.prox_weight}")

    if args.adapt_from is not None:
        # strict=False: base checkpoints predate the adapter and simply have no residual/
        # residual_gate entries, which is exactly the intended starting point (they keep their
        # zero init, so the model is bit-identical to the base). Anything ELSE missing would mean
        # a genuine architecture mismatch, so those are reported rather than swallowed.
        state = torch.load(args.adapt_from, map_location=device, weights_only=False)['state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        adapter_prefixes = ('core.ode_func.residual.', 'core.ode_func.residual_gate.')
        unrelated = [k for k in missing if not k.startswith(adapter_prefixes)]
        if unrelated or unexpected:
            raise RuntimeError(
                "checkpoint does not match this architecture -- missing (non-adapter): "
                f"{unrelated}; unexpected: {list(unexpected)}")

        n_frozen = 0
        for name, p in model.named_parameters():
            if not name.startswith(adapter_prefixes):
                p.requires_grad_(False)
                n_frozen += p.numel()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Adapting from {args.adapt_from}")
        print(f"  frozen base parameters : {n_frozen:,}")
        print(f"  trainable adapter      : {n_train:,}")

    log_path = "logs/" + args.alias + "_gilode_" + args.data + "_" + str(args.sample_percent_train) + "_" + args.mode + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    # The single global scalar alpha (which needed a boosted LR to escape starvation, see
    # CHANGES.md) is gone -- replaced by a per-node state-dependent gate (lib/gil_ode.py), which
    # doesn't share that failure mode, so no special param group is needed anymore.
    #
    # Only trainable parameters are handed to the optimizer: under --adapt-from the base is frozen
    # and weight decay would otherwise still pull frozen weights, since AdamW's decay does not go
    # through .grad and applies to every parameter in its groups.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.l2)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.l2)

    # T_max was 1000 while training runs 50 epochs, so the cosine covered only the first 5% of its
    # period and the learning rate was effectively CONSTANT at 5e-4 for the whole run (5e-4 ->
    # 4.97e-4). Springs extrapolation showed what that costs: validation reaches its minimum
    # between epochs 19 and 38 and then degrades monotonically (+36% to +75% by epoch 50) in every
    # seed, so the model spends the back half taking full-size steps around a minimum it has
    # already passed, and which epoch happens to be best is largely luck -- the direct cause of
    # that cell's seed variance (1.979 / 2.255 / 2.261). Annealing over the actual run length is
    # the standard fix and is uniform across every dataset and both tasks.
    # NOTE: T_max=1000 against a 50-epoch run means the cosine covers only 5% of its period, so the
    # learning rate is effectively constant at 5e-4 (5e-4 -> 4.97e-4). That is almost certainly not
    # what was intended, but annealing properly (T_max=niters, eta_min=1e-6) was tested on springs
    # extrapolation across all three seeds and REJECTED: 2.141 / 2.059 / 2.229 against the paired
    # originals 1.979 / 2.255 / 2.261, i.e. it compressed the spread (range 0.282 -> 0.170) and
    # nudged the mean (2.165 -> 2.143) but made the best seed materially worse and left no seed
    # under the 2.021 bar. It also did not address the underlying problem: validation still peaks
    # early (epochs 17/24/24, vs 32/19/38 before) and degrades afterwards, so springs-extrap is
    # overfitting rather than merely taking too-large steps late. Kept as-is so every reported
    # number stays comparable; revisit only alongside a fix for the overfitting itself.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)

    best_val_mse = np.inf
    best_ckpt_path = None

    # Validation-triggered LR reduction with rollback (--lr-plateau). Springs extrapolation finds
    # its validation minimum between epochs 19 and 38 and then degrades monotonically (+36% to
    # +75% by epoch 50) in every seed, so plain reduction would lower the LR only after the model
    # has already wandered off. Rolling back to the best weights first means the smaller LR
    # refines around the minimum that was actually found, rather than wherever the model drifted.
    # Patience is 10, not 3. Springs-extrap validation gets WORSE before it gets better -- it rises
    # from 0.0324 at epoch 1 to a peak of ~0.0555 around epoch 4 and does not beat its epoch-1
    # value again until epoch 9-10, before descending to its real minimum at epoch 19-38. With
    # patience 3 the rule fires three times inside that normal early transient and burns every LR
    # cut by epoch 10 while rolled back to epoch-1 weights, leaving the model frozen at 4e-6 and
    # unable to ever reach the minimum it was meant to refine (measured, not predicted). Patience
    # must exceed the transient while still catching the post-minimum drift, which spans 12+
    # epochs in every seed.
    PLATEAU_PATIENCE, PLATEAU_FACTOR, PLATEAU_MAX_CUTS, FINAL_PATIENCE = 10, 0.2, 3, 8
    plateau_state = {'since_improve': 0, 'cuts': 0, 'best_sd': None}

    diag_state = {}

    def train_single_batch(batch_dict_encoder, batch_dict_decoder, batch_dict_graph):
        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph)
        loss = train_res["loss"]
        if prox_ref is not None and args.prox_weight > 0:
            prox = sum(((p - prox_ref[n]) ** 2).sum()
                       for n, p in model.named_parameters() if n in prox_ref)
            loss = loss + args.prox_weight * prox
        loss.backward()
        # clip_grad_norm_ returns the pre-clip total norm; capture rather than recompute.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        diag_state['grad_norm'] = max(diag_state.get('grad_norm', 0.0), float(gnorm))
        # Accumulate solver refinements over TRAINING batches. model.core.last_diag is overwritten
        # by the validation and test forward passes that run after each epoch, so reading it in
        # the epoch logger reports an eval batch, not training -- which is why it read 0.
        _d = getattr(model.core, 'last_diag', {})
        diag_state['refines'] = diag_state.get('refines', 0) + _d.get('refines', 0)
        optimizer.step()
        loss_value = loss.data.item()
        del loss
        torch.cuda.empty_cache()
        return loss_value, train_res["mse"], train_res["likelihood"]

    def train_epoch(epo):
        model.train()
        loss_list, mse_list, likelihood_list = [], [], []
        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):
            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, mse, likelihood = train_single_batch(batch_dict_encoder, batch_dict_decoder, batch_dict_graph)

            loss_list.append(loss), mse_list.append(mse), likelihood_list.append(likelihood)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
            torch.cuda.empty_cache()

        if not args.lr_plateau:   # plateau mode drives the LR itself
            scheduler.step()

        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f}|'.format(
            epo, np.mean(loss_list), np.mean(mse_list), np.mean(likelihood_list))

        return message_train

    # Calibrate the stability guard's radius over the first epoch, which is comfortably inside the
    # normal regime (the springs-extrap blowup consistently appears around epoch 4), then freeze
    # it. The guard is inactive while guard_R < 0, so epoch 1 trains exactly as before.
    model.core.calibrating = True

    if args.refine_from is not None or args.forecast_adapter_from is not None:
        # Score the loaded checkpoint before touching it, and seed best_val with that. Refinement
        # is then selected only if it strictly beats the model it started from -- so this
        # procedure cannot return something worse than the existing result.
        model.core.calibrating = False   # radius already calibrated in the loaded checkpoint
        model.eval()
        base_val = compute_loss_all_batches(model, val_encoder, val_graph, val_decoder,
                                            n_batches=val_batch, device=device,
                                            n_traj_samples=1, kl_coef=0.)["mse"]
        best_val_mse = base_val
        best_ckpt_path = args.refine_from or args.forecast_adapter_from
        logger.info('Epoch 0000 [Refine baseline] | val_mse {:.6f} (loaded checkpoint)|'.format(base_val))
        print('refine baseline val mse', base_val)

    for epo in range(1, args.niters + 1):
        message_train = train_epoch(epo)

        if model.core.calibrating:
            radius = model.core.finalize_guard_radius()
            logger.info('Epoch {:04d} [Stability guard] | radius R = {}|'.format(
                epo, 'not calibrated' if radius is None else '{:.4f}'.format(radius)))

        model.eval()
        val_res = compute_loss_all_batches(model, val_encoder, val_graph, val_decoder,
                                           n_batches=val_batch, device=device, n_traj_samples=1, kl_coef=0.)
        # Diagnostic only, never used for checkpoint selection -- see run_models_corrected.py.
        test_res = compute_loss_all_batches(model, test_encoder, test_graph, test_decoder,
                                            n_batches=test_batch, device=device, n_traj_samples=1, kl_coef=0.)

        message_val = 'Epoch {:04d} [Val seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f}|'.format(
            epo, val_res["loss"], val_res["mse"], val_res["likelihood"])
        message_test = 'Epoch {:04d} [Test seq, diagnostic only] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f}|'.format(
            epo, test_res["loss"], test_res["mse"], test_res["likelihood"])

        lam_val = torch.nn.functional.softplus(model.core.lifting.log_lambda).data.item()
        rho_val = torch.nn.functional.softplus(model.core.lifting.log_rho).data.item()
        # Decoder conditioning drives the anchor gain K = W^T (W W^T + ridge)^-1, and a degenerate
        # W has twice killed a run outright (CHANGES.md Part 26). Logged every epoch so drift is
        # visible as it happens rather than reconstructed from a traceback.
        with torch.no_grad():
            W_dec = model.core.decoder.weight
            sv = torch.linalg.svdvals(W_dec)
            cond_val = (sv.max() / sv.min().clamp(min=1e-12)).item()
        # Full instability panel. cond alone is insufficient: it is scale-invariant, so a
        # uniformly shrinking decoder keeps cond flat while sigma_min -> 0 and the anchor gain
        # ||K|| ~ 1/sigma_min explodes. sigma_min and ||K|| are logged explicitly for that reason.
        with torch.no_grad():
            sv_min = float(sv.min())
        d = getattr(model.core, 'last_diag', {})
        logger.info(('Epoch {:04d} [Instability] | sigma_min {:.4e} | cond {:.2f} | ||K|| {:.4f} | '
                     'corr_max {:.4f} | h_final {:.4f} | h_max {:.4f} | ||f|| {:.4f} | '
                     'grad_norm {:.4e} | refines {}|').format(
            epo, sv_min, cond_val, d.get('K_norm', float('nan')), d.get('corr_max', float('nan')),
            d.get('h_final', float('nan')), d.get('h_max', float('nan')),
            d.get('f_norm', float('nan')), diag_state.get('grad_norm', float('nan')),
            diag_state.get('refines', -1)))
        diag_state['grad_norm'] = 0.0
        diag_state['refines'] = 0

        gamma_val = torch.nn.functional.softplus(model.core.ode_func.log_gamma).data.item()
        message_graph = ('Epoch {:04d} [Graph params] | lambda {:.4f} | rho {:.4f} | '
                         'decoder cond {:.2f} | guard gamma {:.4f} R {:.4f}|').format(
            epo, lam_val, rho_val, cond_val, gamma_val, float(model.core.ode_func.guard_R))
        if args.particles > 1:
            message_graph += ' diffusion sigma mean {:.4f}|'.format(
                torch.nn.functional.softplus(model.core.log_diffusion).mean().item())
        if args.reversion:
            message_graph += ' reversion gate mean {:.4f}|'.format(getattr(model.core, 'last_diag_rev', float('nan')))

        logger.info("Experiment " + str(experimentID))
        logger.info(message_train)
        logger.info(message_val)
        logger.info(message_test)
        logger.info(message_graph)
        print("data: %s, model: GIL-ODE, sample: %s, mode:%s" % (args.data, str(args.sample_percent_train), args.mode))

        cur_lr = optimizer.param_groups[0]['lr']
        logger.info('Epoch {:04d} [LR] | lr {:.6e}|'.format(epo, cur_lr))

        if val_res["mse"] < best_val_mse:
            best_val_mse = val_res["mse"]
            # Recorded so the schedule's effect can be read off directly: which epoch was selected
            # and what the learning rate was there.
            logger.info('Epoch {:04d} [Selected] | val_mse {:.6f} | lr {:.6e}|'.format(
                epo, val_res["mse"], cur_lr))
            message_best = 'Epoch {:04d} [Val seq (cond on sampled tp)] | Best val mse {:.6f}|'.format(epo, best_val_mse)
            logger.info(message_best)
            best_ckpt_path = os.path.join(args.save, "experiment_" + str(
                experimentID) + "_gilode_" + args.data + "_" + str(
                args.sample_percent_train) + "_" + args.mode + "_epoch_" + str(epo) + "_valmse_" + str(
                best_val_mse) + '.ckpt')
            torch.save({'args': args, 'state_dict': model.state_dict()}, best_ckpt_path)
            if args.lr_plateau:
                plateau_state['since_improve'] = 0
                plateau_state['best_sd'] = {k: v.detach().clone()
                                            for k, v in model.state_dict().items()}
        elif args.lr_plateau:
            plateau_state['since_improve'] += 1
            at_final_lr = plateau_state['cuts'] >= PLATEAU_MAX_CUTS
            patience = FINAL_PATIENCE if at_final_lr else PLATEAU_PATIENCE

            if plateau_state['since_improve'] >= patience:
                if at_final_lr:
                    logger.info(('Epoch {:04d} [Plateau] | no improvement for {:d} epochs at the '
                                 'final LR -- stopping|').format(epo, FINAL_PATIENCE))
                    print('early stop at epoch', epo)
                    break
                # Roll back to the best-validation weights BEFORE cutting the LR, so the smaller
                # step refines around the minimum that was found rather than the drifted state.
                if plateau_state['best_sd'] is not None:
                    model.load_state_dict(plateau_state['best_sd'])
                new_lr = cur_lr * PLATEAU_FACTOR
                # Rebuild the optimizer so Adam's moment estimates, accumulated while drifting
                # away from the minimum, do not carry across the rollback.
                opt_cls = optim.AdamW if args.optimizer == "AdamW" else optim.Adam
                optimizer = opt_cls([p for p in model.parameters() if p.requires_grad],
                                    lr=new_lr, weight_decay=args.l2)
                plateau_state['cuts'] += 1
                plateau_state['since_improve'] = 0
                logger.info(('Epoch {:04d} [Plateau] | rolled back to best val {:.6f} | '
                             'lr {:.3e} -> {:.3e} | cut {:d}/{:d}|').format(
                    epo, best_val_mse, cur_lr, new_lr, plateau_state['cuts'], PLATEAU_MAX_CUTS))

        torch.cuda.empty_cache()

    utils.get_ckpt_model(best_ckpt_path, model, device)
    model.eval()
    final_test_res = compute_loss_all_batches(model, test_encoder, test_graph, test_decoder,
                                              n_batches=test_batch, device=device, n_traj_samples=1, kl_coef=0.)
    message_final = 'FINAL (best-val checkpoint) [Test seq] | MSE {:.6f} | Likelihood {:.6f}|'.format(
        final_test_res["mse"], final_test_res["likelihood"])
    logger.info(message_final)
    print(message_final)

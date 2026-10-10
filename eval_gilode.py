'''
Evaluation-only tool for trained GIL-ODE checkpoints. Nothing here trains or selects; every
number comes from a fixed best-validation checkpoint.

It rebuilds the exact data split the checkpoint was trained on (same args, same seed, same
loader), runs the core model directly, and reports:

  mse_full    the standard metric every baseline reports (VAE_Baseline.get_mse: per-sequence,
              per-feature time-average, then the mean), averaged over batches exactly as
              lib.utils.compute_loss_all_batches does
  mse_prefix  the training wrapper's extrapolation metric (mean over 20..100% horizon prefixes,
              batch-pooled), i.e. what run_models_gilode.py's FINAL line reports in extrap mode
  per-step    squared error and counts per decoder time index (error-vs-horizon curves)

Test-time interventions (no retraining):
  --causal            switch the backward smoother off (causal filter, interpolation)
  --drop-agents k     fully mask k agents' observations in the conditioning window, per
                      trajectory (sensor outage); metrics are also reported on those agents only
  --knockout-lift     zero the lifted correction to unobserved agents (delta_U = 0)
  --graph G           replace the support S: true | complete | empty | rewire (same edge count)
  --dump FILE.npz     save predictions, targets, masks and times for figures
'''
import argparse
import json
import os
import sys

import numpy as np
import torch

import lib.utils as utils
from lib.baseline_gil_ode import GILODEBaseline, HORIZON_FRACTIONS, _pooled_masked_mse
from lib.gil_dataset import prepare_gil_batch

p = argparse.ArgumentParser('GIL-ODE evaluation')
p.add_argument('--ckpt', required=True)
p.add_argument('--split', default='test', choices=['test', 'val'])
p.add_argument('--causal', action='store_true')
p.add_argument('--decode-prior', action='store_true', help='predict each target from the state before its own observation is applied')
p.add_argument('--drop-agents', type=int, default=0)
p.add_argument('--drop-seed', type=int, default=0)
p.add_argument('--knockout-lift', action='store_true')
p.add_argument('--graph', default='true', choices=['true', 'complete', 'empty', 'rewire'])
p.add_argument('--lift-stats', action='store_true', help='record ||beta*delta_U|| / ||r_O|| per event')
p.add_argument('--conductance', default=None, help='save the mean learned conductance w_ij over all events to this .npy')
p.add_argument('--dump', default=None)
p.add_argument('--dump-batches', type=int, default=1)
p.add_argument('--out', default=None, help='append the JSON result line to this file')
p.add_argument('--tag', default='')
p.add_argument('--particles', type=int, default=None, help='override the ensemble size at test time')
p.add_argument('--batch-size', type=int, default=None, help='override the eval batch size (memory)')
cli = p.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ck = torch.load(cli.ckpt, map_location=device, weights_only=False)
args = ck['args']

if cli.batch_size is not None:
    args.batch_size = cli.batch_size
# Same seeding as training, so the loader draws the identical observation masks.
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
if args.data == 'ieee39':
    from lib.ieee39_dataLoader import IEEE39ParseData
    dataloader = IEEE39ParseData(args.dataset, mode=args.mode, args=args)
else:
    from lib.corrected_dataLoader import CorrectedParseData
    dataloader = CorrectedParseData(args.dataset, suffix=args.suffix, mode=args.mode, args=args)
loaded = {}
for split in ('train', 'val', 'test'):
    sp = args.sample_percent_train if split == 'train' else args.sample_percent_test
    loaded[split] = dataloader.load_data(sample_percent=sp, batch_size=args.batch_size, data_type=split)
enc, dec, graph, n_batches = loaded[cli.split]

model = GILODEBaseline(input_dim=dataloader.feature, hidden_dim=args.hidden_dim, num_atoms=args.n_balls,
                       dataset=args.data, obsrv_std=torch.Tensor([0.01]).to(device), device=device,
                       mode=args.mode, mlp_width=args.mlp_width, ode_substeps=args.ode_substeps,
                       ode_tol=args.ode_tol, use_smoother=getattr(args, 'smoother', False),
                       smoother_mode=getattr(args, 'smoother_mode', 'coldstart'),
                       ablate=tuple(a for a in getattr(args, 'ablate', '').split(',') if a),
                       use_reversion=getattr(args, 'reversion', False),
                       n_particles=cli.particles or getattr(args, 'particles', 1)).to(device)
model.load_state_dict(ck['state_dict'], strict=False)
model.eval()
core = model.core
if cli.causal:
    core.use_smoother = False
core.decode_prior = cli.decode_prior

N = args.n_balls
rng = torch.Generator(device='cpu').manual_seed(cli.drop_seed)

# ---- interventions on the lifting step ----
lift_log = {'ratio_sum': 0.0, 'n': 0}
_orig_lift = core.lifting.forward
_orig_gate = core.gate.forward


cond_acc = {'sum': None, 'n': 0}


def lifting_hook(h_minus, S, c, r, mask):
    delta = _orig_lift(h_minus, S, c, r, mask)
    if cli.conductance:
        L = core.lifting
        B_, N_, H_ = h_minus.shape
        hi = h_minus.unsqueeze(2).expand(B_, N_, N_, H_)
        hj = h_minus.unsqueeze(1).expand(B_, N_, N_, H_)
        w = S * torch.nn.functional.softplus(L.g(torch.cat([hi, hj, c.unsqueeze(-1)], -1)).squeeze(-1))
        w = 0.5 * (w + w.transpose(1, 2))
        cond_acc['sum'] = w.sum(0) if cond_acc['sum'] is None else cond_acc['sum'] + w.sum(0)
        cond_acc['n'] += B_
    if cli.knockout_lift:
        delta = mask.unsqueeze(-1) * delta
    lifting_hook.last = (delta, r, mask)
    return delta


def gate_hook(tso, dens, dnorm):
    beta = _orig_gate(tso, dens, dnorm)
    if cli.lift_stats and getattr(lifting_hook, 'last', None) is not None:
        delta, r, mask = lifting_hook.last
        unobs = (1 - mask)
        num = ((beta.squeeze(-1) * delta.norm(dim=-1)) * unobs).sum(1) / unobs.sum(1).clamp(min=1)
        den = (r.norm(dim=-1) * mask).sum(1) / mask.sum(1).clamp(min=1)
        ok = (unobs.sum(1) > 0) & (den > 0)
        lift_log['ratio_sum'] += (num[ok] / den[ok]).sum().item()
        lift_log['n'] += int(ok.sum())
    return beta


core.lifting.forward = lifting_hook
core.gate.forward = gate_hook


def perturb_graph(S):
    if cli.graph == 'true':
        return S
    B = S.shape[0]
    eye = torch.eye(N, device=S.device).unsqueeze(0)
    if cli.graph == 'complete':
        return (1 - eye).expand(B, -1, -1).clone()
    if cli.graph == 'empty':
        return torch.zeros_like(S)
    # rewire: a random symmetric graph with the same number of edges, per trajectory
    out = torch.zeros_like(S)
    iu = torch.triu_indices(N, N, 1)
    for b in range(B):
        m = int(S[b][iu[0], iu[1]].sum().item())
        pick = torch.randperm(iu.shape[1], generator=rng)[:m]
        out[b, iu[0, pick], iu[1, pick]] = 1
    return out + out.transpose(1, 2)


def per_seq_mse(pred, truth, mask):
    '''[M, T, D] each -> [M] per-sequence MSE, identical to lib.likelihood_eval.compute_masked_likelihood
    with the mse function (per-feature time-average, then mean over features). NaN where a
    sequence has no target.'''
    se = ((pred - truth) ** 2 * mask).sum(1)            # [M, D]
    cnt = mask.sum(1)                                    # [M, D]
    return (se / cnt).mean(-1)


tot = {'mse_full': 0.0, 'mse_prefix': 0.0, 'mse_check': 0.0, 'mse_dropped': [], 'mse_kept': []}
step_se, step_cnt = None, None
dumps = []
with torch.no_grad():
    for bi in range(n_batches):
        b_en = utils.get_next_batch_new(enc, device)
        b_gr = utils.get_next_batch_new(graph, device)
        b_de = utils.get_next_batch(dec, device)
        dense, mask, grid_times, S, c = prepare_gil_batch(b_en, b_de, b_gr, N, args.data, device)
        B = dense.shape[0]

        dropped = torch.zeros(B, N, dtype=torch.bool, device=device)
        if cli.drop_agents > 0:
            for b in range(B):
                dropped[b, torch.randperm(N, generator=rng)[:cli.drop_agents].to(device)] = True
            mask = mask & ~dropped.unsqueeze(-1)

        S_used = perturb_graph(S)
        if args.data == 'charged' and cli.graph != 'true':
            pass  # charged S is already complete; 'complete' is a no-op there by construction
        pred, _, _ = core(dense, mask, grid_times, b_de['time_steps'], S_used, c)
        T_Q = pred.shape[2]
        pred_flat = pred.reshape(B * N, T_Q, -1)
        truth, tmask = b_de['data'], b_de['mask']

        qi = torch.searchsorted(grid_times, b_de['time_steps'])
        seen = mask[:, :, qi].reshape(B * N, T_Q)                       # target time observed for that agent
        tgt = tmask.any(-1)
        tot.setdefault('overlap', []).append((seen & tgt).sum().item() / max(tgt.sum().item(), 1))
        # Split the standard metric by whether the target point was also a conditioning input.
        # In interpolation 'seen' targets can be reproduced by an exact-anchoring model; only the
        # 'unseen' ones measure interpolation proper. Per-sequence then nanmean (a sequence with no
        # target in a group is skipped), matching get_mse's normalization within each group.
        sm = seen.unsqueeze(-1).float()
        for name, mk in (('mse_seen', tmask * sm), ('mse_unseen', tmask * (1 - sm))):
            tot.setdefault(name, []).append(torch.nanmean(per_seq_mse(pred_flat, truth, mk)).item())
        tot['mse_check'] += model.get_mse(truth, pred_flat.unsqueeze(0), mask=tmask).item()
        seq = per_seq_mse(pred_flat, truth, tmask)
        tot['mse_full'] += seq.mean().item()
        cutoffs = sorted(set(max(1, round(T_Q * f)) for f in HORIZON_FRACTIONS)) if args.mode == 'extrap' else [T_Q]
        tot['mse_prefix'] += torch.stack([_pooled_masked_mse(pred_flat.unsqueeze(0)[:, :, :H], truth[:, :H], tmask[:, :H])
                                          for H in cutoffs]).mean().item()
        if cli.drop_agents > 0:
            d = dropped.reshape(-1)
            tot['mse_dropped'].append(seq[d].mean().item())
            tot['mse_kept'].append(seq[~d].mean().item())

        if core.n_particles > 1:
            X = core.last_samples                                   # [B, K, N, T_Q, D]
            var_t = (X.var(1, unbiased=True).reshape(B * N, T_Q, -1) * tmask).sum(dim=(0, 2)).cpu().numpy()
            tot['var_step'] = var_t if 'var_step' not in tot else tot['var_step'] + var_t
        se_t = ((pred_flat - truth) ** 2 * tmask).sum(dim=(0, 2)).cpu().numpy()
        cnt_t = tmask.sum(dim=(0, 2)).cpu().numpy()
        step_se = se_t if step_se is None else step_se + se_t
        step_cnt = cnt_t if step_cnt is None else step_cnt + cnt_t

        if cli.dump and bi < cli.dump_batches:
            dumps.append(dict(pred=pred.cpu().numpy(), truth=truth.reshape(B, N, T_Q, -1).cpu().numpy(),
                              tmask=tmask.reshape(B, N, T_Q, -1).cpu().numpy(),
                              t_query=b_de['time_steps'].cpu().numpy(), dense=dense.cpu().numpy(),
                              obs_mask=mask.cpu().numpy(), grid_times=grid_times.cpu().numpy(),
                              S=S_used.cpu().numpy(), dropped=dropped.cpu().numpy()))

tot['overlap'] = float(np.mean(tot.get('overlap', [0])))
split = {k: float(np.nanmean(tot[k])) for k in ('mse_seen', 'mse_unseen') if k in tot}
res = {'tag': cli.tag, 'target_obs_overlap': tot['overlap'], 'ckpt': os.path.basename(cli.ckpt), 'data': args.data, 'mode': args.mode,
       'seed': args.random_seed, 'split': cli.split, 'causal': cli.causal, 'drop_agents': cli.drop_agents,
       'knockout_lift': cli.knockout_lift, 'graph': cli.graph, 'decode_prior': cli.decode_prior,
       'mse_full': tot['mse_full'] / n_batches, 'mse_check': tot['mse_check'] / n_batches,
       'mse_prefix': tot['mse_prefix'] / n_batches,
       **split, 'particles': core.n_particles,
       'var_step': (tot['var_step'] / np.maximum(step_cnt, 1)).tolist() if 'var_step' in tot else None,
       'mse_step': (step_se / np.maximum(step_cnt, 1)).tolist()}
if cli.drop_agents > 0:
    res['mse_dropped'] = float(np.mean(tot['mse_dropped']))
    res['mse_kept'] = float(np.mean(tot['mse_kept']))
if cli.lift_stats:
    res['lift_ratio'] = lift_log['ratio_sum'] / max(lift_log['n'], 1)
line = json.dumps(res)
print('RESULT ' + json.dumps({k: v for k, v in res.items() if k != 'mse_step'}))
if cli.out:
    with open(cli.out, 'a') as f:
        f.write(line + '\n')
if cli.conductance:
    np.save(cli.conductance, (cond_acc['sum'] / max(cond_acc['n'], 1)).cpu().numpy())
if cli.dump:
    os.makedirs(os.path.dirname(cli.dump) or '.', exist_ok=True)
    # Batches can have different grid lengths; keep only arrays whose shapes agree across batches.
    keys = [k for k in dumps[0] if all(d[k].shape[1:] == dumps[0][k].shape[1:] for d in dumps)]
    np.savez_compressed(cli.dump, **{k: np.concatenate([d[k] for d in dumps]) if dumps[0][k].ndim and k not in ('t_query', 'grid_times')
                                     else dumps[0][k] for k in keys})

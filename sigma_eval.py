'''
Antithetic sigma-flow forecast mean (CHANGES.md Part 41), evaluation only.

At the last observation the state is split exactly into decoder-visible and decoder-null parts
(QR projector, the FNC hook in lib/gil_ode.py). Three deterministic forecasts are rolled out from
  h_0 = h_row + h_null,  h_+ = h_row + (1 + delta) h_null,  h_- = h_row + (1 - delta) h_null,
all with the same decoded boundary state, and combined as
  y = (1 - 2w) y_0 + w y_+ + w y_-.
The +/- pair cancels the first-order effect of the perturbation and keeps the second-order
(curvature) correction to the forecast mean. delta = 0 reproduces the baseline exactly.

For each batch the baseline forecast is computed once and each delta's pair once, then every
(delta, w) combination is scored from those shared forecasts. Standard metric (per-sequence,
per-feature time average, then mean), averaged over batches as compute_loss_all_batches does.
'''
import argparse, json, sys
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--split', default='val', choices=['val', 'test'])
p.add_argument('--deltas', type=float, nargs='+', default=[0.05, 0.10, 0.20, 0.30])
p.add_argument('--weights', type=float, nargs='+', default=[0.25, 0.50])
p.add_argument('--out', required=True)
scli = p.parse_args()
sys.argv = ['x', '--ckpt', scli.ckpt]
src = open('eval_gilode.py').read().split("N = args.n_balls")[0]
exec(src)
from lib.gil_dataset import prepare_gil_batch
assert args.mode == 'extrap'
N = args.n_balls


def per_seq(pred, truth, m):
    se = ((pred - truth) ** 2 * m).sum(1)
    return torch.nanmean((se / m.sum(1)).mean(-1)).item()


def step_mse(pred, truth, m):
    return (((pred - truth) ** 2 * m).sum((0, 2)) / m.sum((0, 2)).clamp(min=1)).cpu().numpy()


combos = [(0.0, 0.0)] + [(d, w) for d in scli.deltas for w in scli.weights]
acc = {c: [] for c in combos}
steps = {c: [] for c in combos}
spread = {d: [] for d in scli.deltas}
enc, dec, gr, nb = loaded[scli.split]
with torch.no_grad():
    for bi in range(nb):
        be = utils.get_next_batch_new(enc, device); bg = utils.get_next_batch_new(gr, device); bd = utils.get_next_batch(dec, device)
        dense, mask, grid, S, c = prepare_gil_batch(be, bd, bg, N, args.data, device)
        B = dense.shape[0]
        truth, tm = bd['data'], bd['mask']

        def fc(alpha):
            core.fnc_alpha = alpha
            out = core(dense, mask, grid, bd['time_steps'], S, c)[0]
            core.fnc_alpha = None
            return out.reshape(B * N, out.shape[2], -1)

        y0 = fc(None)
        pairs = {d: (fc(1.0 + d), fc(1.0 - d)) for d in scli.deltas}
        for (d, w) in combos:
            y = y0 if d == 0.0 else (1 - 2 * w) * y0 + w * pairs[d][0] + w * pairs[d][1]
            acc[(d, w)].append(per_seq(y, truth, tm))
            steps[(d, w)].append(step_mse(y, truth, tm))
        for d in scli.deltas:
            yp, ym = pairs[d]
            ybar = 0.5 * (yp + ym)
            sp = 0.5 * (((yp - ybar) ** 2 + (ym - ybar) ** 2) * tm).sum((0, 2)) / tm.sum((0, 2)).clamp(min=1)
            spread[d].append(sp.cpu().numpy())

with open(scli.out, 'a') as f:
    for (d, w) in combos:
        f.write(json.dumps({'ckpt': scli.ckpt.split('/')[-1], 'seed': args.random_seed, 'split': scli.split,
                            'delta': d, 'weight': w, 'mse_full': float(np.mean(acc[(d, w)])),
                            'mse_step': np.mean(steps[(d, w)], 0).tolist(),
                            'spread_step': (np.mean(spread[d], 0).tolist() if d > 0 else None)}) + '\n')
print('done', scli.split, args.random_seed)

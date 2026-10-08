'''
Evaluation-only tool for any baseline checkpoint (Corrected LG-ODE, ODE-RNN, Latent-ODE, Edge-GNN,
RNN-NRI, CSG-ODE). It reuses the baseline's own run script for data and model construction --
everything above that script's `log_path =` line -- with the checkpoint's saved args, so the split,
masks and architecture are exactly the ones it was trained and selected on. Every baseline's
compute_all_losses calls VAE_Baseline.get_mse(truth, pred, mask); that call is intercepted to
record per-decoder-step squared error, which is all the horizon figure needs. The reported
mse_full is the unmodified metric, so it must reproduce the FINAL line of the training log.

  python eval_baseline.py --script run_models_corrected.py --ckpt experiments/<file>.ckpt \
      --model LG-ODE --out run_logs/eval/baseline_steps.jsonl
'''
import argparse
import json
import os
import sys

import numpy as np
import torch

p = argparse.ArgumentParser('baseline evaluation')
p.add_argument('--script', required=True)
p.add_argument('--ckpt', required=True)
p.add_argument('--model', required=True, help='display name, e.g. LG-ODE')
p.add_argument('--out', default=None)
cli = p.parse_args()

src = open(cli.script).read()
head, sep, rest = src.partition('args = parser.parse_args()')
assert sep, 'run script has no parse_args line'
body = rest.split('    log_path =')[0]

ck_args = torch.load(cli.ckpt, map_location='cpu', weights_only=False)['args']
ck_args.save = os.path.dirname(cli.ckpt) + '/'
ck_args.load = os.path.basename(cli.ckpt)

ns = {'__name__': '__main__', '__file__': cli.script}
sys.argv = [cli.script]
exec(compile(head + sep, cli.script, 'exec'), ns)
# Checkpoints saved before a flag existed lack it; fill from the script's own defaults (for
# --csg-solver that is 'euler', which is what those runs used).
for k, v in vars(ns['parser'].parse_args([])).items():
    if not hasattr(ck_args, k):
        setattr(ck_args, k, v)
ns['args'] = ck_args
exec(compile(body, cli.script, 'exec'), ns)        # builds data, model and loads the checkpoint
model, device = ns['model'], ns['device']
test_encoder, test_graph, test_decoder, test_batch = (ns['test_encoder'], ns['test_graph'],
                                                     ns['test_decoder'], ns['test_batch'])

acc = {'se': None, 'cnt': None, 'seen': [], 'unseen': []}
_orig = model.get_mse

# Remember the current batch, so get_mse can tell which targets were also conditioning inputs
# (see eval_gilode.py: in interpolation ~59% are). lib.utils.compute_loss_all_batches looks these
# helpers up on the module at call time, so wrapping the module attributes is enough.
import lib.utils as utils
from lib.nri_baseline import build_dense_grid
_cur = {}
_gnb_new, _gnb = utils.get_next_batch_new, utils.get_next_batch
def _wrap_new(it, dev):
    b = _gnb_new(it, dev)
    if hasattr(b, 'pos'):
        _cur['enc'] = b
    return b
def _wrap(it, dev):
    b = _gnb(it, dev)
    _cur['dec'] = b
    return b
utils.get_next_batch_new, utils.get_next_batch = _wrap_new, _wrap


def seq_mse(pred, truth, mask):
    return ((pred - truth) ** 2 * mask).sum(1).div(mask.sum(1)).mean(-1)


def get_mse_hook(truth, pred_y, mask=None):
    with torch.no_grad():
        se = ((pred_y[0] - truth) ** 2 * mask).sum(dim=(0, 2)).cpu().numpy()
        cnt = mask.sum(dim=(0, 2)).cpu().numpy()
        acc['se'] = se if acc['se'] is None else acc['se'] + se
        acc['cnt'] = cnt if acc['cnt'] is None else acc['cnt'] + cnt
        en, tq = _cur['enc'], _cur['dec']['time_steps']
        _, omask, grid = build_dense_grid(en.x, en.pos, en.y, tq, truth.device)
        seen = omask[:, torch.searchsorted(grid, tq)].unsqueeze(-1)          # [M, T_Q, 1]
        acc['seen'].append(torch.nanmean(seq_mse(pred_y[0], truth, mask * seen)).item())
        acc['unseen'].append(torch.nanmean(seq_mse(pred_y[0], truth, mask * (1 - seen))).item())
    return _orig(truth, pred_y, mask=mask)


model.get_mse = get_mse_hook
model.eval()
res = utils.compute_loss_all_batches(model, test_encoder, test_graph, test_decoder, n_batches=test_batch,
                                     device=device, n_traj_samples=1, kl_coef=0.)
out = {'model': cli.model, 'ckpt': os.path.basename(cli.ckpt), 'data': ck_args.data, 'mode': ck_args.mode,
       'seed': ck_args.random_seed, 'mse_full': res['mse'],
       'mse_seen': float(np.nanmean(acc['seen'])), 'mse_unseen': float(np.nanmean(acc['unseen'])),
       'mse_step': (acc['se'] / np.maximum(acc['cnt'], 1)).tolist()}
print('RESULT ' + json.dumps({k: v for k, v in out.items() if k != 'mse_step'}))
if cli.out:
    with open(cli.out, 'a') as f:
        f.write(json.dumps(out) + '\n')

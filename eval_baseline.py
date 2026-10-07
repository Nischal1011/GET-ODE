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
ns['args'] = ck_args
exec(compile(body, cli.script, 'exec'), ns)        # builds data, model and loads the checkpoint
model, device = ns['model'], ns['device']
test_encoder, test_graph, test_decoder, test_batch = (ns['test_encoder'], ns['test_graph'],
                                                     ns['test_decoder'], ns['test_batch'])

acc = {'se': None, 'cnt': None}
_orig = model.get_mse


def get_mse_hook(truth, pred_y, mask=None):
    with torch.no_grad():
        se = ((pred_y[0] - truth) ** 2 * mask).sum(dim=(0, 2)).cpu().numpy()
        cnt = mask.sum(dim=(0, 2)).cpu().numpy()
        acc['se'] = se if acc['se'] is None else acc['se'] + se
        acc['cnt'] = cnt if acc['cnt'] is None else acc['cnt'] + cnt
    return _orig(truth, pred_y, mask=mask)


model.get_mse = get_mse_hook
import lib.utils as utils
model.eval()
res = utils.compute_loss_all_batches(model, test_encoder, test_graph, test_decoder, n_batches=test_batch,
                                     device=device, n_traj_samples=1, kl_coef=0.)
out = {'model': cli.model, 'ckpt': os.path.basename(cli.ckpt), 'data': ck_args.data, 'mode': ck_args.mode,
       'seed': ck_args.random_seed, 'mse_full': res['mse'],
       'mse_step': (acc['se'] / np.maximum(acc['cnt'], 1)).tolist()}
print('RESULT ' + json.dumps({k: v for k, v in out.items() if k != 'mse_step'}))
if cli.out:
    with open(cli.out, 'a') as f:
        f.write(json.dumps(out) + '\n')

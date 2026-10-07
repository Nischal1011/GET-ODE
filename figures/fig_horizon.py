'''
Error vs forecast horizon (extrapolation), one panel per dataset, mean +- std over seeds.

Inputs are JSONL files from eval_gilode.py / eval_baseline.py, each passed as PATH:LABEL; every
row needs data, mode, seed and mse_step (per-decoder-step MSE).

  python figures/fig_horizon.py run_logs/eval/rescore_extrap.jsonl:GIL-ODE \
      run_logs/eval/baseline_steps.jsonl: --out paper/figures/horizon.pdf
(an empty LABEL keeps each row's own 'model' field)
'''
import argparse

import numpy as np

import style
from common import read_jsonl, mean_std

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='+')
ap.add_argument('--out', default='paper/figures/horizon.pdf')
ap.add_argument('--log', action='store_true')
ap.add_argument('--bins', type=int, default=20, help='average per-step MSE into this many horizon bins')
a = ap.parse_args()

rows = []
for spec in a.inputs:
    path, _, label = spec.partition(':')
    for r in read_jsonl(path):
        r['model'] = label or r.get('model') or r.get('tag')
        rows.append(r)
rows = [r for r in rows if r['mode'] == 'extrap' and 'mse_step' in r]

style.setup()
datasets = [d for d in style.DATASETS if any(r['data'] == d for r in rows)]
fig, axes = style.plt.subplots(1, len(datasets), figsize=(2.2 * len(datasets), 1.8), squeeze=False)
for ax, d in zip(axes[0], datasets):
    for m in dict.fromkeys(r['model'] for r in rows if r['data'] == d):
        curves = [np.asarray(r['mse_step']) * 100 for r in rows if r['data'] == d and r['model'] == m]
        L = min(len(c) for c in curves)
        nb = min(a.bins, L)
        edges = np.linspace(0, L, nb + 1).astype(int)
        mu, sd = mean_std([[c[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])] for c in curves])
        x = (edges[:-1] + edges[1:]) / (2 * L)
        ax.plot(x, mu, color=style.color(m), lw=1.2, label=m)
        ax.fill_between(x, mu - sd, mu + sd, color=style.color(m), alpha=0.2, lw=0)
    ax.set_title(style.DATASETS[d])
    ax.set_xlabel('forecast horizon (fraction)')
    if a.log:
        ax.set_yscale('log')
axes[0, 0].set_ylabel(r'MSE ($\times 10^{-2}$)')
axes[0, -1].legend(frameon=False, loc='upper left')
style.save(fig, a.out)

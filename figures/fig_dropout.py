'''
Agent-dropout / sensor-outage curve (the thesis figure): MSE on the masked agents only vs the
number of masked agents k, one panel per dataset/task, mean +- std over seeds.

Rows (PATH:LABEL as in fig_horizon.py) need data, mode, seed, drop_agents and mse_dropped. k = 0
rows have no masked agents; their mse_full is plotted as the reference point.
'''
import argparse

import numpy as np

import style
from common import read_jsonl, mean_std

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs='+')
ap.add_argument('--mode', default='interp')
ap.add_argument('--out', default='paper/figures/dropout.pdf')
a = ap.parse_args()

rows = []
for spec in a.inputs:
    path, _, label = spec.partition(':')
    for r in read_jsonl(path):
        r['model'] = label or r.get('model') or r.get('tag')
        rows.append(r)
rows = [r for r in rows if r['mode'] == a.mode]

style.setup()
datasets = [d for d in style.DATASETS if any(r['data'] == d for r in rows)]
fig, axes = style.plt.subplots(1, len(datasets), figsize=(2.2 * len(datasets), 1.8), squeeze=False)
for ax, d in zip(axes[0], datasets):
    for m in dict.fromkeys(r['model'] for r in rows if r['data'] == d):
        ks = sorted({r['drop_agents'] for r in rows if r['data'] == d and r['model'] == m})
        mus, sds = [], []
        for k in ks:
            vals = [(r['mse_dropped'] if k > 0 else r['mse_full']) * 100
                    for r in rows if r['data'] == d and r['model'] == m and r['drop_agents'] == k]
            mu, sd = mean_std(vals)
            mus.append(mu); sds.append(sd)
        ax.errorbar(ks, mus, yerr=sds, color=style.color(m), marker='o', ms=3, lw=1.2, capsize=2, label=m)
    ax.set_title(f"{style.DATASETS[d]} ({a.mode})")
    ax.set_xlabel('masked agents $k$')
    ax.set_yscale('log')
axes[0, 0].set_ylabel(r'MSE on masked agents ($\times 10^{-2}$)')
axes[0, -1].legend(frameon=False)
style.save(fig, a.out)

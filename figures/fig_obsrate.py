'''
Error vs observed fraction (0.4 / 0.6 / 0.8), one panel per dataset and task.

Input: a CSV with columns model,data,mode,rate,seed,mse (MSE in raw units), produced by
run_logs/collect_obsrate.py from the training logs.
'''
import argparse
import csv
from collections import defaultdict

import numpy as np

import style

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--out', default='paper/figures/obsrate.pdf')
a = ap.parse_args()

vals = defaultdict(list)
with open(a.csv) as f:
    for r in csv.DictReader(f):
        vals[(r['data'], r['mode'], r['model'], float(r['rate']))].append(float(r['mse']) * 100)

style.setup()
cells = sorted({(d, m) for d, m, _, _ in vals}, key=lambda x: (list(style.DATASETS).index(x[0]), x[1]))
fig, axes = style.plt.subplots(1, len(cells), figsize=(1.8 * len(cells), 1.8), squeeze=False)
for ax, (d, mode) in zip(axes[0], cells):
    for model in dict.fromkeys(k[2] for k in vals if k[:2] == (d, mode)):
        rates = sorted(k[3] for k in vals if k[:3] == (d, mode, model))
        mu = [np.mean(vals[(d, mode, model, r)]) for r in rates]
        sd = [np.std(vals[(d, mode, model, r)], ddof=1) if len(vals[(d, mode, model, r)]) > 1 else 0 for r in rates]
        ax.errorbar(rates, mu, yerr=sd, color=style.color(model), marker='o', ms=3, lw=1.2, capsize=2, label=model)
    ax.set_title(f'{style.DATASETS[d]} {mode}')
    ax.set_xlabel('observed fraction')
    ax.set_yscale('log')
axes[0, 0].set_ylabel(r'MSE ($\times 10^{-2}$)')
axes[0, -1].legend(frameon=False, fontsize=6)
style.save(fig, a.out)

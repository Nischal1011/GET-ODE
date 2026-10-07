'''
Learned lifting conductances vs the physical Kron-reduced admittance on IEEE39 (Fig. 8).

Inputs: one or more .npy files of mean conductance w_ij from
  python eval_gilode.py --ckpt <ieee39 checkpoint> --conductance w_s1991.npy
(averaged over seeds if several), and data/ieee39_kron_reduced.npz. Plots the two heatmaps and a
scatter over the support edges with Spearman's rho. Note the model only sees the binary support S,
never the admittance values, so any correlation is learned.
'''
import argparse

import numpy as np
from scipy.stats import spearmanr

import style

ap = argparse.ArgumentParser()
ap.add_argument('w', nargs='+')
ap.add_argument('--kron', default='data/ieee39_kron_reduced.npz')
ap.add_argument('--out', default='paper/figures/conductance.pdf')
a = ap.parse_args()

W = np.mean([np.load(p) for p in a.w], axis=0)
K = np.load(a.kron)
Y = np.abs(K['y_reduced_real'] + 1j * K['y_reduced_imag'])
np.fill_diagonal(Y, 0)
edges = np.triu(W > 0, 1)
rho, pval = spearmanr(Y[edges], W[edges])

style.setup()
fig, ax = style.plt.subplots(1, 3, figsize=(6.6, 2.0), gridspec_kw={'width_ratios': [1, 1, 1.1]})
for axi, M, t in [(ax[0], Y * (W > 0), r'$|Y_{\mathrm{Kron}}|$ on support'), (ax[1], W, r'learned $w_{ij}$')]:
    im = axi.imshow(M, cmap='viridis')
    axi.set_title(t)
    axi.set_xticks(range(10)); axi.set_yticks(range(10))
    axi.set_xticklabels([f'G{i+1}' for i in range(10)], rotation=90, fontsize=5)
    axi.set_yticklabels([f'G{i+1}' for i in range(10)], fontsize=5)
    fig.colorbar(im, ax=axi, fraction=0.046, pad=0.04)
ax[2].scatter(Y[edges], W[edges], s=10, color=style.color('GIL-ODE'))
ax[2].set_xlabel(r'$|Y_{\mathrm{Kron}}|_{ij}$')
ax[2].set_ylabel(r'learned $w_{ij}$')
ax[2].set_title(f'Spearman $\\rho$ = {rho:.2f} (p = {pval:.1g}, {edges.sum()} edges)')
print(f'spearman rho={rho:.3f} p={pval:.3g} over {edges.sum()} support edges')
style.save(fig, a.out)

'''
Qualitative rollouts (appendix): 2-D trajectories of every agent for one test trajectory, truth vs
prediction, from eval_gilode.py --dump files (one per model, PATH:LABEL). Features 0/1 are taken as
the 2-D position (springs/charged). For IEEE39 use --features to pick (e.g. rotor angle, speed).
'''
import argparse

import numpy as np

import style

ap = argparse.ArgumentParser()
ap.add_argument('dumps', nargs='+')
ap.add_argument('--traj', type=int, default=0)
ap.add_argument('--features', type=int, nargs=2, default=[0, 1])
ap.add_argument('--out', default='paper/figures/qualitative.pdf')
a = ap.parse_args()

style.setup()
fig, axes = style.plt.subplots(1, len(a.dumps), figsize=(2.0 * len(a.dumps), 2.0), squeeze=False)
fx, fy = a.features
for ax, spec in zip(axes[0], a.dumps):
    path, _, label = spec.partition(':')
    Z = np.load(path)
    P, T, M = Z['pred'][a.traj], Z['truth'][a.traj], Z['tmask'][a.traj]
    cols = style.plt.cm.tab10(np.arange(P.shape[0]) % 10)
    for i in range(P.shape[0]):
        m = M[i, :, fx] > 0
        ax.plot(T[i, m, fx], T[i, m, fy], color=cols[i], lw=1.0)
        ax.plot(P[i, :, fx], P[i, :, fy], color=cols[i], lw=1.0, ls='--')
        ax.scatter(T[i, m, fx][:1], T[i, m, fy][:1], color=cols[i], s=6)
    ax.set_title(label or path)
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].plot([], [], 'k-', label='truth'); axes[0, 0].plot([], [], 'k--', label='prediction')
axes[0, 0].legend(frameon=False)
style.save(fig, a.out)

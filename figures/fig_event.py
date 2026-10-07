'''
Event-level explanation (Fig. 3): what one observation does to the agent that was observed and
to its unobserved neighbours, with lifting vs without (test-time knockout, same checkpoint, same
batch).

Inputs: two dumps of the same checkpoint from eval_gilode.py,
  --dump full.npz                      (normal model)
  --dump nolift.npz --knockout-lift    (delta_U = 0 at test time)
Picks the trajectory/event where the lifted correction moved an unobserved neighbour most (or use
--traj/--event), and plots feature --feat over time for the observed agent and its two
strongest-coupled neighbours: truth (black), GIL-ODE (blue), no lifting (cyan), observations (dots).
'''
import argparse

import numpy as np

import style

ap = argparse.ArgumentParser()
ap.add_argument('full')
ap.add_argument('nolift')
ap.add_argument('--traj', type=int, default=None)
ap.add_argument('--event', type=int, default=None, help='grid index of the event')
ap.add_argument('--feat', type=int, default=0)
ap.add_argument('--out', default='paper/figures/event.pdf')
a = ap.parse_args()

F, X = np.load(a.full), np.load(a.nolift)
pred, pred0 = F['pred'], X['pred']                    # [B, N, TQ, D]
truth, tmask, tq = F['truth'], F['tmask'], F['t_query']
obs, grid, S = F['obs_mask'], F['grid_times'], F['S']  # [B, N, T], [T], [B, N, N]
B, N = obs.shape[:2]

# The figure is only honest where the neighbour actually has targets to compare against: score
# every (trajectory, single-agent event) by the improvement lifting buys on the neighbours, and
# show the best one, as the caption must then say ("selected example").
if a.traj is None or a.event is None:
    best = (-np.inf, 0, 0)
    gain = ((pred0 - truth) ** 2 - (pred - truth) ** 2)[..., a.feat] * tmask[..., a.feat]   # [B, N, TQ]
    for b in range(B):
        for k in np.where(obs[b].sum(0) == 1)[0]:
            o = int(np.where(obs[b, :, k])[0][0])
            nb = np.where(S[b, o] > 0)[0]
            after = (tq > grid[k]) & (tq <= grid[k] + 0.15 * (tq[-1] - tq[0]))
            g = gain[b][np.ix_(nb, after)].sum() if len(nb) and after.any() else -np.inf
            if g > best[0]:
                best = (g, b, k)
    _, b, k = best
else:
    b, k = a.traj, a.event
o = int(np.where(obs[b, :, k])[0][0])
nb = [int(j) for j in np.argsort(-S[b, o])[:2] if S[b, o, j] > 0]

style.setup()
agents = [o] + nb
fig, axes = style.plt.subplots(1, len(agents), figsize=(2.1 * len(agents), 1.7), sharex=True)
for ax, i in zip(np.atleast_1d(axes), agents):
    m = tmask[b, i, :, a.feat] > 0
    ax.plot(tq[m], truth[b, i, m, a.feat], color='black', lw=1.0, label='truth')
    ax.plot(tq, pred0[b, i, :, a.feat], color=style.color('GIL-ODE (no lifting)'), lw=1.0, ls='--', label='no lifting')
    ax.plot(tq, pred[b, i, :, a.feat], color=style.color('GIL-ODE'), lw=1.2, label='GIL-ODE')
    ob = obs[b, i] > 0
    ax.scatter(grid[ob], F['dense'][b, i, ob, a.feat], s=8, color='black', zorder=5)
    ax.axvline(grid[k], color='tab:red', lw=0.8, ls=':')
    ax.set_title(('observed agent %d' if i == o else 'unobserved neighbour %d') % i)
    ax.set_xlabel('time')
np.atleast_1d(axes)[0].legend(frameon=False, loc='best')
print(f'trajectory {b}, event index {k} (t={grid[k]:.3f}), observed agent {o}, neighbours {nb}')
style.save(fig, a.out)

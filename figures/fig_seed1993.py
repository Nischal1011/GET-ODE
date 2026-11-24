'''Why springs-extrapolation seed 1993 fails (CHANGES.md Part 50). Three panels from the seed study
(run_logs/eval/seed_study/{study,channels}.json; late-horizon validation MSE x1e-2):
 (a) late error over training: the failed run forks away from the healthy seeds at epochs 3-6;
 (b) late error by number of springs: healthy runs improve with more coupling, the failed run degrades;
 (c) switching off one interaction channel at a time: healthy runs depend on the antisymmetric
     channel, the failed run on the free-form messages.'''
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from style import setup, save

HEALTHY, FAILED = '#1f77b4', '#D55E00'   # GIL-ODE blue; vermilion (CVD-validated pair)

p = argparse.ArgumentParser()
p.add_argument('--study', default='run_logs/eval/seed_study/study.json')
p.add_argument('--channels', default='run_logs/eval/seed_study/channels.json')
p.add_argument('--out', default='paper/figures/seed1993.pdf')
a = p.parse_args()
S, C = json.load(open(a.study)), json.load(open(a.channels))

setup()
fig, ax = plt.subplots(1, 3, figsize=(6.75, 1.95), gridspec_kw={'width_ratios': [1, 1, 1.35]})

# (a) training trajectory
def traj(prefix):
    pts = sorted((int(t.split('_e')[1]), r['late']) for t, r in S.items() if t.startswith(prefix + '_e'))
    return zip(*pts)
for prefix, lab in (('h91', 'seed 1991'), ('h92', 'seed 1992')):
    x, y = traj(prefix)
    ax[0].plot(x, y, '-o', color=HEALTHY, lw=1.2, ms=3, alpha=0.85)
x, y = traj('fail')
ax[0].plot(x, y, '-o', color=FAILED, lw=1.2, ms=3)
ax[0].axvspan(3, 6, color='0.9', zorder=0, lw=0)
ax[0].text(4.5, 17.5, 'fork', ha='center', va='top', fontsize=6.5, color='0.35')
ax[0].set(xlabel='epoch', ylabel='late-horizon val. MSE (×1e-2)', ylim=(0, 18), title='(a) training')
ax[0].text(17.5, 15.2, 'seed 1993', color=FAILED, fontsize=6.5, ha='left')
ax[0].text(48, 0.4, 'seeds 1991, 1992', color=HEALTHY, fontsize=6.5, ha='right')

# (b) error by number of springs (1-9; trajectories with no springs are ~10 for every model)
ks = [str(k) for k in range(1, 10)]
for t in ('h91_e50', 'h92_e40', 'h94_best', 'h95_best'):
    ax[1].plot(range(1, 10), [S[t]['err_by_nsprings'][k] for k in ks], color=HEALTHY, lw=1.0, alpha=0.7)
ax[1].plot(range(1, 10), [S['fail_e10']['err_by_nsprings'][k] for k in ks], color=FAILED, lw=1.6)
ax[1].plot(range(1, 10), [S['fail_e3']['err_by_nsprings'][k] for k in ks], color=FAILED, lw=1.0, ls='--')
ax[1].text(9, 24.5, 'seed 1993, ep. 10', color=FAILED, fontsize=6.5, ha='right')
ax[1].text(9, 11.6, 'ep. 3 (selected)', color=FAILED, fontsize=6.5, ha='right')
ax[1].text(9, 3.4, 'four healthy seeds', color=HEALTHY, fontsize=6.5, ha='right')
ax[1].set(xlabel='number of springs in the system', ylim=(0, 30), xticks=[1, 3, 5, 7, 9], title='(b) by coupling')

# (c) channel switch-off
def iv(t):
    d = dict(S[t].get('interv', {})); d.update(C.get(t, {}).get('interv', {})); return d
conds = [('baseline', 'intact'), ('no_antisym', 'no\nantisym.'), ('no_free_messages', 'no free\nmsgs'),
         ('interaction_gate_x0.5', 'gate\n×0.5'), ('interaction_off', 'no\ninteract.')]
hs = ['h91_e50', 'h92_e40', 'h94_best']
H = np.array([[iv(t)[k] for k, _ in conds] for t in hs]); F = np.array([iv('fail_e10')[k] for k, _ in conds])
xi, w = np.arange(len(conds)), 0.36
ax[2].bar(xi - w / 2 - 0.01, H.mean(0), w, color=HEALTHY, label='healthy (3 seeds)')
for row in H:
    ax[2].plot(xi - w / 2 - 0.01, row, 'o', ms=2.2, color='white', mec=HEALTHY, mew=0.6, zorder=3)
ax[2].bar(xi + w / 2 + 0.01, F, w, color=FAILED, label='seed 1993, ep. 10')
ax[2].set_xticks(xi, [l for _, l in conds], fontsize=6)
ax[2].set(ylim=(0, 23), yticks=[0, 5, 10, 15], title='(c) channel switch-off')
ax[2].legend(frameon=False, loc='upper center', ncol=2, fontsize=6.5, handlelength=1.0, columnspacing=1.0)
for axx in ax:
    axx.grid(axis='y', color='0.92', lw=0.6); axx.set_axisbelow(True)
fig.tight_layout(w_pad=1.0)
save(fig, a.out)

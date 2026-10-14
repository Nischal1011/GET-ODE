'''Bias-variance split of test MSE across seeds (charged extrap): for each model, the mean
single-seed MSE, the MSE of the 3-seed prediction average, and their difference (the part of the
error due to seed-to-seed disagreement of the learned model). Evaluation only.'''
import numpy as np
def per_seq(p, y, m):
    p, y, m = p.reshape(-1, *p.shape[-2:]), y.reshape(-1, *y.shape[-2:]), m.reshape(-1, *m.shape[-2:])
    return np.nanmean((((p - y) ** 2 * m).sum(1) / m.sum(1)).mean(-1))
for lab, pat in (('GIL-ODE (det)', 'run_logs/eval/full_charged_s{}_test.npz'),
                 ('LG-ODE', 'run_logs/eval/lg_charged_s{}_test.npz')):
    Z = [np.load(pat.format(s)) for s in (1991, 1992, 1993)]
    same = all(np.array_equal(Z[0]['truth'], z['truth']) for z in Z)
    singles = [per_seq(z['pred'], z['truth'], z['tmask']) * 100 for z in Z]
    ens = per_seq(np.mean([z['pred'] for z in Z], 0), Z[0]['truth'], Z[0]['tmask']) * 100
    print(f'{lab:<14} identical targets across seeds: {same} | single {np.round(singles,3)} mean {np.mean(singles):.3f} '
          f'| 3-seed average {ens:.3f} | seed-disagreement share {np.mean(singles)-ens:.3f} ({(np.mean(singles)-ens)/np.mean(singles):.0%})')

'''Five-seed charged-extrap comparison (CHANGES.md Part 47): raw, sigma-flow, and affine lead-time
recalibration fitted on each model's own validation forecasts. Test, x1e-2.'''
import numpy as np
O = 'run_logs/eval/mos/'; S = (1991, 1992, 1993, 1994, 1995)
def L(name, s, sp, key):
    d = np.load(f'{O}{name}_s{s}_{sp}.npz'); return d[key].astype(np.float64), d['truth'].astype(np.float64), d['tmask'].astype(np.float64)
mse = lambda p, t, m: np.nanmean(((p - t) ** 2 * m).sum(1) / m.sum(1)) * 100
def fit(p, t, m):
    n = m.sum(0); pm = (p * m).sum(0) / n; tm = (t * m).sum(0) / n
    a = (((p - pm) * (t - tm)) * m).sum(0) / (((p - pm) ** 2) * m).sum(0); return a, tm - a * pm
rows = [('GIL MoE-4 (50 ep)', 'gilo', 'pred0'), ('GIL MoE-4 (50 ep) + sigma-flow', 'gilo', 'pred'), ('LG-ODE (50 ep)', 'lg', 'pred'),
        ('GIL MoE-4 + continuation', 'gil', 'pred0'), ('GIL MoE-4 + continuation + sigma-flow', 'gil', 'pred'), ('LG-ODE + continuation', 'lgc', 'pred')]
for lab, name, key in rows:
    raw, cal = [], []
    for s in S:
        pv, tv, mv = L(name, s, 'val', key); pt, tt, mt = L(name, s, 'test', key); a, b = fit(pv, tv, mv)
        raw.append(mse(pt, tt, mt)); cal.append(mse(a * pt + b, tt, mt))
    f = lambda x: '%.3f +- %.3f [%s]' % (np.mean(x), np.std(x, ddof=1), ' '.join('%.3f' % v for v in x))
    print('%-38s raw %s | calibrated %s' % (lab, f(raw), f(cal)))

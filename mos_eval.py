'''
Lead-time recalibration (CHANGES.md Part 45), evaluation only, applied identically to every model.
Per forecast step k and feature f, a scalar shrink a[k,f] = <truth, pred> / <pred, pred> is fitted
on the model's own VALIDATION predictions and applied to its TEST predictions: y_hat = a * pred.
Metric: standard per-sequence, per-feature time average, then mean.
'''
import sys, numpy as np
O = 'run_logs/eval/mos/'
def L(name, s, sp, key='pred'):
    d = np.load(f'{O}{name}_s{s}_{sp}.npz'); return d[key].astype(np.float64), d['truth'].astype(np.float64), d['tmask'].astype(np.float64)
def mse(p, t, m):
    return np.nanmean(((p - t) ** 2 * m).sum(1) / m.sum(1)) * 100
def fit(p, t, m, mode):
    if mode == 'slope':
        return (p * t * m).sum(0) / np.maximum((p * p * m).sum(0), 1e-12), 0.0
    n = m.sum(0); pm = (p * m).sum(0) / n; tm = (t * m).sum(0) / n
    a = (((p - pm) * (t - tm)) * m).sum(0) / np.maximum((((p - pm) ** 2) * m).sum(0), 1e-12)
    return a, tm - a * pm
rows = [('GIL mixture+cont', 'gil', 'pred0'), ('GIL mixture+cont+sigma', 'gil', 'pred'), ('LG-ODE original', 'lg', 'pred'), ('LG-ODE continued', 'lgc', 'pred')]
for mode in ('slope', 'affine'):
    print('== calibration:', mode)
    for label, name, key in rows:
        raw, cal, sl = [], [], []
        for s in (1991, 1992, 1993):
            pv, tv, mv = L(name, s, 'val', key); pt, tt, mt = L(name, s, 'test', key)
            a, b = fit(pv, tv, mv, mode)
            raw.append(mse(pt, tt, mt)); cal.append(mse(a * pt + b, tt, mt)); sl.append(a)
        A = np.mean(sl, 0)
        print('%-24s raw %.3f (%s) | calibrated %.3f (%s) | mean slope steps 13-24 %.2f, 25-42 %.2f, 43-60 %.2f | n_val %d' % (
            label, np.mean(raw), ' '.join('%.3f' % x for x in raw), np.mean(cal), ' '.join('%.3f' % x for x in cal),
            A[12:24].mean(), A[24:42].mean(), A[42:].mean(), pv.shape[0]))

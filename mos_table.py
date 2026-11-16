'''
Lead-time recalibration for every model on the extrapolation cells (CHANGES.md Part 46).
Per forecast step and feature, a linear map fitted on the model's own VALIDATION forecasts is
applied to its TEST forecasts. Variant (slope / affine) is chosen by 2-fold cross-validation on
validation data only, pooled over all models and cells. Standard metric, x1e-2.
'''
import glob, json, numpy as np
O = 'run_logs/eval/mos_all/'
def load(f):
    d = np.load(f); p = d['pred'].astype(np.float64); t = d['truth'].astype(np.float64); m = d['tmask'].astype(np.float64)
    return p.reshape(-1, *p.shape[-2:]), t.reshape(-1, *t.shape[-2:]), m.reshape(-1, *m.shape[-2:])
def mse(p, t, m):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nanmean(((p - t) ** 2 * m).sum(1) / m.sum(1)) * 100
def fit(p, t, m, mode):
    if mode == 'slope':
        return (p * t * m).sum(0) / np.maximum((p * p * m).sum(0), 1e-12), 0.0
    n = np.maximum(m.sum(0), 1); pm = (p * m).sum(0) / n; tm = (t * m).sum(0) / n
    a = (((p - pm) * (t - tm)) * m).sum(0) / np.maximum((((p - pm) ** 2) * m).sum(0), 1e-12)
    return a, tm - a * pm
models = ['GIL-ODE', 'LG-ODE', 'CSG-ODE', 'Latent-ODE', 'ODE-RNN', 'Edge-GNN', 'RNN-NRI']
R = {}; cv = {'slope': [], 'affine': [], 'raw': []}
for data in ('spring', 'charged', 'ieee39'):
    for mdl in models:
        for s in (1991, 1992, 1993):
            pv, tv, mv = load(f'{O}{mdl}_{data}_s{s}_val.npz'); pt, tt, mt = load(f'{O}{mdl}_{data}_s{s}_test.npz')
            r = {'raw': mse(pt, tt, mt)}
            h = pv.shape[0] // 2; idx = np.random.RandomState(0).permutation(pv.shape[0]); A, B = idx[:h], idx[h:]
            base = mse(pv, tv, mv)
            for mode in ('slope', 'affine'):
                a, b = fit(pv, tv, mv, mode); r[mode] = mse(a * pt + b, tt, mt)
                e = 0
                for tr, te in ((A, B), (B, A)):
                    a, b = fit(pv[tr], tv[tr], mv[tr], mode); e += 0.5 * mse(a * pv[te] + b, tv[te], mv[te])
                cv[mode].append(e / base)
            R[(data, mdl, s)] = r
print('validation-only 2-fold CV, mean error relative to uncalibrated: slope %.4f affine %.4f' % (np.mean(cv['slope']), np.mean(cv['affine'])))
for data in ('spring', 'charged', 'ieee39'):
    print('==', data, 'extrap (mean of 3 seeds; per-seed in brackets)')
    for mdl in models:
        g = lambda k: [R[(data, mdl, s)][k] for s in (1991, 1992, 1993)]
        print('%-11s raw %8.4f | slope %8.4f [%s] | affine %8.4f [%s]' % (mdl, np.mean(g('raw')), np.mean(g('slope')), ' '.join('%.3f' % x for x in g('slope')), np.mean(g('affine')), ' '.join('%.3f' % x for x in g('affine'))))
json.dump({'|'.join(map(str, k)): v for k, v in R.items()}, open('run_logs/eval/mos_all/table.json', 'w'))

'''Oracle check for horizon-dependent shrinkage toward the mean (evaluation only).
For each extrap checkpoint: fit alpha_t per forecast step on VALIDATION,
  y_hat' = (1 - alpha_t) * y_hat + alpha_t * mu,   mu = per-feature mean of validation targets,
then score TEST with the standard per-sequence metric. Also: averaging the 3 seed models.'''
import numpy as np, glob, os
def per_seq(p, y, m):                       # [B,N,T,D] -> standard get_mse value
    se = ((p - y) ** 2 * m).sum(2); c = m.sum(2)
    return np.nanmean((se / c).mean(-1))
E = 'run_logs/eval'
for d in ('charged', 'spring', 'ieee39'):
    preds = {}
    for s in (1991, 1992, 1993):
        fv, ft = f'{E}/full_{d}_s{s}_val.npz', f'{E}/full_{d}_s{s}_test.npz'
        if not (os.path.exists(fv) and os.path.exists(ft)): continue
        V, T = np.load(fv), np.load(ft)
        pv, yv, mv = V['pred'], V['truth'], V['tmask']
        pt, yt, mt = T['pred'], T['truth'], T['tmask']
        mu = (yv * mv).sum((0, 1, 2)) / mv.sum((0, 1, 2))
        a = ((pv - mu) * (pv - yv) * mv).sum((0, 1, 3)) / (((pv - mu) ** 2) * mv).sum((0, 1, 3))
        a = np.clip(a, 0, 1)[None, None, :, None]
        base, shr = per_seq(pt, yt, mt), per_seq((1 - a) * pt + a * mu, yt, mt)
        preds[s] = (pt, yt, mt)
        print(f'{d} s{s}: test {base*100:.3f} -> shrunk {shr*100:.3f}   alpha by fifth',
              np.round([a[0,0,i:j].mean() for i, j in zip(*[np.linspace(0, a.shape[2], 6).astype(int)[k:] for k in (0, 1)])], 2))
    if len(preds) == 3 and all(np.array_equal(preds[1991][1], preds[s][1]) for s in preds):
        avg = np.mean([preds[s][0] for s in preds], 0)
        print(f'{d} 3-seed prediction average: {per_seq(avg, preds[1991][1], preds[1991][2])*100:.3f}')

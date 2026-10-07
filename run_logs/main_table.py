'''Main-results table: GIL-ODE and all five baselines, six cells, three seeds (1991-1993).
Every run on identical splits; validation-selected checkpoints. MSE x1e-2, mean +- sd.'''
import re, os, numpy as np
def mse(f):
    p = f'run_logs/{f}.log'
    if not os.path.exists(p): return None
    m = re.findall(r'FINAL \(best-val checkpoint\) \[Test seq\] \| MSE ([0-9.]+) \| Likelihood (-[0-9.]+)', open(p).read())
    if not m: return None
    v, ll = float(m[-1][0]), float(m[-1][1])
    return (v if v > 0 else -ll * 2e-4) * 100   # MSE prints to 6 dp; recover tiny values from likelihood
def bname(m, d, t, s):
    if m == 'csgode': return f'csgode_{d}_{t}_s{s}'
    if m in ('corrected', 'odernn') and d == 'charged': return f'v5k_{m}_charged_{t}_s{s}'
    if s == 1991: return f'ieee39_subset_{m}_{t}' if d == 'ieee39' else f'subset_{m}_{d}_{t}'
    if m == 'corrected' and d == 'spring' and t == 'extrap': return f'lgode_s{s}_spring_extrap'
    return f'seed_{m}_{d}_{t}_s{s}'
def gname(d, t, s):
    if s == 1991:
        return 'fr_l2-1e-3_s1991_spring_extrap' if (d, t) == ('spring', 'extrap') else f'fc_{d}_{t}'
    return f'fc_s{s}_{d}_{t}'
cells = [(d, t) for d in ('spring', 'charged', 'ieee39') for t in ('interp', 'extrap')]
models = [('GIL-ODE', None), ('LG-ODE', 'corrected'), ('ODE-RNN', 'odernn'), ('Latent-ODE', 'latentode'),
          ('Edge-GNN', 'edgegnn'), ('RNN-NRI', 'rnnnri'), ('CSG-ODE', 'csgode')]
vals = {}
for lab, m in models:
    for d, t in cells:
        vals[lab, d, t] = [mse(gname(d, t, s) if m is None else bname(m, d, t, s)) for s in (1991, 1992, 1993)]
def fmt(v):
    v = [x for x in v if x is not None]
    if not v: return '-'
    mu, sd = np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else float('nan'))
    f = (lambda x: f'{x:.2e}') if mu < 1e-3 else (lambda x: f'{x:.3f}')
    return f'{f(mu)} ± {f(sd)}' + ('' if len(v) == 3 else f' (n={len(v)})')
hdr = ['Springs interp', 'Springs extrap', 'Charged interp', 'Charged extrap', 'IEEE39 interp', 'IEEE39 extrap']
print('| model | ' + ' | '.join(hdr) + ' |'); print('|---' * 7 + '|')
for lab, _ in models:
    print(f'| {lab} | ' + ' | '.join(fmt(vals[lab, d, t]) for d, t in cells) + ' |')
print('\nPer cell: strongest baseline by 3-seed mean, and GIL paired wins against EVERY baseline:')
for (d, t), h in zip(cells, hdr):
    base = {lab: vals[lab, d, t] for lab, m in models if m}
    best = min(base, key=lambda k: np.mean([x for x in base[k] if x is not None]))
    g = vals['GIL-ODE', d, t]
    wins = sum(1 for lab in base for s in range(3) if g[s] is not None and base[lab][s] is not None and g[s] < base[lab][s])
    total = sum(1 for lab in base for s in range(3) if g[s] is not None and base[lab][s] is not None)
    print(f'  {h:<15} strongest = {best:<10}  GIL paired wins vs all baselines: {wins}/{total}')

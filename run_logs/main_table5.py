'''Five-seed main-results table (seeds 1991-1995): GIL-ODE final config (MoE-4, run_logs/final5.sh)
and all six baselines, six cells. Identical splits across seeds; validation-selected checkpoints;
raw test MSE x1e-2 from each run's FINAL line (standard full-horizon metric for every model).
Sigma-flow (extrap) is reported separately from sigma_eval.py outputs, see CHANGES.md.'''
import re, os, sys, numpy as np
SEEDS = (1991, 1992, 1993, 1994, 1995)
def mse(f):
    p = f'run_logs/{f}.log'
    if not os.path.exists(p): return None
    m = re.findall(r'FINAL \(best-val checkpoint\) \[Test seq\] \| MSE ([0-9.]+) \| Likelihood (-[0-9.]+)', open(p).read())
    if not m: return None
    v, ll = float(m[-1][0]), float(m[-1][1])
    return (v if v > 0 else -ll * 2e-4) * 100   # MSE prints to 6 dp; recover tiny values from likelihood
def bnames(m, d, t, s):
    c = []
    if m == 'csgode': c.append(f'csgode_{d}_{t}_s{s}')
    if m in ('corrected', 'odernn') and d == 'charged': c.append(f'v5k_{m}_charged_{t}_s{s}')
    if s == 1991: c.append(f'ieee39_subset_{m}_{t}' if d == 'ieee39' else f'subset_{m}_{d}_{t}')
    if m == 'corrected' and d == 'spring' and t == 'extrap': c.append(f'lgode_s{s}_spring_extrap')
    c += [f'seed_{m}_{d}_{t}_s{s}', f's5_{m}_{d}_{t}_s{s}']
    return c
def first(names):
    for n in names:
        v = mse(n)
        if v is not None: return v, n
    return None, None
cells = [(d, t) for d in ('spring', 'charged', 'ieee39') for t in ('interp', 'extrap')]
models = [('GIL-ODE', None), ('LG-ODE', 'corrected'), ('ODE-RNN', 'odernn'), ('Latent-ODE', 'latentode'),
          ('Edge-GNN', 'edgegnn'), ('RNN-NRI', 'rnnnri'), ('CSG-ODE', 'csgode')]
vals, src = {}, {}
for lab, m in models:
    for d, t in cells:
        r = [first([f'moe_s{s}_{d}_{t}'] if m is None else bnames(m, d, t, s)) for s in SEEDS]
        vals[lab, d, t] = [x[0] for x in r]; src[lab, d, t] = [x[1] for x in r]
if '--sources' in sys.argv:
    for k, v in src.items(): print(k, v)
def fmt(v):
    v = [x for x in v if x is not None]
    if not v: return '-'
    mu, sd = np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else float('nan'))
    f = (lambda x: f'{x:.2e}') if mu < 1e-3 else (lambda x: f'{x:.3f}')
    return f'{f(mu)} ± {f(sd)}' + ('' if len(v) == len(SEEDS) else f' (n={len(v)})')
hdr = ['Springs interp', 'Springs extrap', 'Charged interp', 'Charged extrap', 'IEEE39 interp', 'IEEE39 extrap']
print('| model | ' + ' | '.join(hdr) + ' |'); print('|---' * 7 + '|')
for lab, _ in models:
    print(f'| {lab} | ' + ' | '.join(fmt(vals[lab, d, t]) for d, t in cells) + ' |')
print('\nPer cell: strongest baseline (5-seed mean), GIL per-seed values, paired wins vs strongest and vs all:')
for (d, t), h in zip(cells, hdr):
    base = {lab: vals[lab, d, t] for lab, m in models if m}
    best = min(base, key=lambda k: np.mean([x for x in base[k] if x is not None]))
    g = vals['GIL-ODE', d, t]
    pairs = lambda b: [(gs, bs) for gs, bs in zip(g, b) if gs is not None and bs is not None]
    wb = sum(gs < bs for gs, bs in pairs(base[best]))
    wa = sum(gs < bs for lab in base for gs, bs in pairs(base[lab])); ta = sum(len(pairs(base[lab])) for lab in base)
    fs = lambda v: ' '.join('-' if x is None else (f'{x:.2e}' if x < 1e-3 else f'{x:.3f}') for x in v)
    print(f'  {h:<15} best={best:<10} GIL [{fs(g)}] best [{fs(base[best])}]  wins vs best {wb}/5, vs all {wa}/{ta}')

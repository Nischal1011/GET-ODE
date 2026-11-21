import json
R = json.load(open('run_logs/eval/seed_study/study.json'))
print(f"{'tag':22s} {'late':>6s} {'velL5':>6s} {'top10':>5s} {'rhoDeg':>6s} {'null s/m/e':>16s} {'>R':>5s} {'g':>5s} {'gInt':>5s} {'fSelf':>5s} {'res':>5s} {'gateH':>5s} {'disag':>5s} {'J absc':>6s} {'J imag':>6s}")
for t, r in R.items():
    d = r['decomp']; mean = lambda k: sum(d[k]) / len(d[k]) if k in d else float('nan')
    print(f"{t:22s} {r['late']:6.2f} {r['fifths_vel'][-1]:6.2f} {r['top10_share']:5.2f} {r['rho_err_degree']:6.2f} "
          f"{'/'.join(f'{x:.2f}' for x in r['null_norm_start_mid_end']):>16s} {r['beyond_guard_frac']:5.2f} "
          f"{mean('g'):5.2f} {mean('g_int'):5.2f} {mean('f_self'):5.2f} {mean('res'):5.2f} {mean('gate_H'):5.2f} {mean('exp_disagree'):5.2f} "
          f"{r['jac']['mid']['abscissa']:6.2f} {r['jac']['mid']['max_imag']:6.2f}")
print('\nerror by number of springs (late MSE x1e-2):')
for t, r in R.items():
    print(f"{t:22s} " + ' '.join(f"{k}:{v:5.1f}" for k, v in r['err_by_nsprings'].items()))
print('\ninterventions (late MSE x1e-2):')
for t, r in R.items():
    if 'interv' in r:
        print(f"{t:22s} " + ' '.join(f"{k}={v:.2f}" for k, v in r['interv'].items()))

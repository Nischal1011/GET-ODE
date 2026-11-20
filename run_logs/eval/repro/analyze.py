'''Part 48 diagnosis: what goes wrong in the failed springs-extrap mixture run (repro s1993)?
Per fifth of the forecast horizon, on validation: MSE split position/velocity, amplitude ratio
std(pred)/std(truth) (>1 = over-dispersed / energy growth), regression slope truth~pred, correlation,
and phase drift proxy (best lag of the cross-correlation of each trajectory).'''
import numpy as np, sys, os
O = 'run_logs/eval/repro'
tags = [t for t in ['fail_e2', 'fail_e3', 'fail_e6', 'fail_e10', 'fail_e15', 'h91_e2', 'h91_e10', 'h91_e13', 'h91_e20', 'h91_e50',
                    'h92_e2', 'h92_e10', 'h92_e20', 'h92_e40'] if os.path.exists(f'{O}/{t}.npz')]
for t in tags:
    d = np.load(f'{O}/{t}.npz')
    p, y, m = d['pred'], d['truth'], d['tmask']           # [B, N, T, D]
    T, D = p.shape[2], p.shape[3]
    fifths = np.array_split(np.arange(T), 5)
    row = []
    for f in fifths:
        pp, yy, mm = p[:, :, f], y[:, :, f], m[:, :, f] > 0
        mse_pos = ((pp - yy) ** 2)[..., :D // 2][mm[..., :D // 2]].mean()
        mse_vel = ((pp - yy) ** 2)[..., D // 2:][mm[..., D // 2:]].mean()
        a, b = pp[mm], yy[mm]
        amp = a.std() / b.std(); slope = (a * b).sum() / (a * a).sum(); corr = np.corrcoef(a, b)[0, 1]
        row.append(f'{100*mse_pos:6.2f}/{100*mse_vel:6.2f} amp {amp:4.2f} slope {slope:4.2f} r {corr:4.2f}')
    print(f'{t:9s} | ' + ' || '.join(row))
    # phase-drift proxy on the last two fifths: per trajectory/agent/feature, lag (in steps, -6..6) maximising overlap
    late = np.concatenate(fifths[3:])
    lags = []
    for k in range(-6, 7):
        sl = slice(max(0, k), len(late) + min(0, k)); sl2 = slice(max(0, -k), len(late) + min(0, -k))
        lags.append(((p[:, :, late][:, :, sl] - y[:, :, late][:, :, sl2]) ** 2).mean(axis=(2, 3)))
    best = np.array(lags).argmin(0) - 6
    print(f'{"":9s}   late-horizon best lag: mean {best.mean():+.2f}, |lag|>=2 in {np.mean(np.abs(best) >= 2):.0%} of agent-trajectories')

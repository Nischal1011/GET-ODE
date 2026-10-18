'''
Test-time procedures on a trained GIL-ODE extrapolation checkpoint (evaluation only; CHANGES.md
Part 37). Standard metric on the standard test split.
  lagged : lagged-average forecasting (Hoffman & Kalnay 1983) -- average the uncorrected forecasts
           started from the filter's posterior state at the last L observation events (every
           `stride` events), all verifying at the same query times.
  4dvar  : strong-constraint 4D-Var -- optimise an increment to the posterior state at the start
           of an assimilation window covering the last `window` fraction of the context so that
           the UNCORRECTED model trajectory fits every observation in the window (plus a weak
           background term), then forecast from that trajectory.
'''
import argparse, json, sys
import numpy as np
import torch
p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--lags', type=int, nargs='*', default=[2, 4, 6])
p.add_argument('--stride', type=int, default=2)
p.add_argument('--windows', type=float, nargs='*', default=[0.25, 0.5])
p.add_argument('--iters', type=int, default=30)
p.add_argument('--lr', type=float, default=0.05)
p.add_argument('--bg', type=float, default=1e-3)
p.add_argument('--batches', type=int, default=4)
p.add_argument('--out', default=None)
tcli = p.parse_args()
sys.argv = ["x", "--ckpt", tcli.ckpt]
src = open('eval_gilode.py').read().split("N = args.n_balls")[0]
exec(src)
from torchdiffeq import odeint as odeint_direct
from lib.gil_dataset import prepare_gil_batch
N = args.n_balls

def per_seq(pred, truth, m):
    se = ((pred - truth) ** 2 * m).sum(1); c = m.sum(1)
    return torch.nanmean((se / c).mean(-1))

def tau_at(obs_mask, grid, j):
    '''time since each agent's last observation at grid index j'''
    T = grid.shape[0]
    idx = torch.arange(T, device=grid.device).view(1, 1, T)
    last = torch.where(obs_mask[:, :, :j + 1], idx[:, :, :j + 1], torch.full_like(idx[:, :, :j + 1], -1)).amax(-1)
    t_last = torch.where(last >= 0, grid[last.clamp(min=0)], grid[0] - 10.0)
    return (grid[j] - t_last).clamp(max=10.0)

def rollout(h, grid, j, tau, exact=True):
    out = [h]
    for k in range(j, grid.shape[0] - 1):
        t0, t1 = grid[k], grid[k + 1]
        if (t1 - t0).abs() > 1e-8:
            core.ode_func.set_tau(tau, t0)
            h = core._integrate(h, t0, t1) if (exact and core.ode_tol is not None) else \
                odeint_direct(core.ode_func, h, torch.stack([t0, t1]), method='rk4')[-1]
            tau = tau + (t1 - t0)
        out.append(h)
    return torch.stack(out, 2)            # [B, N, T - j, H]

res = {'base': [], **{f'lag{L}': [] for L in tcli.lags}, **{f'4dvar{w}': [] for w in tcli.windows}}
enc, dec, gr, nb = loaded['test']
for bi in range(min(tcli.batches, nb)):
    be = utils.get_next_batch_new(enc, device); bg = utils.get_next_batch_new(gr, device); bd = utils.get_next_batch(dec, device)
    dense, mask, grid, S, c = prepare_gil_batch(be, bd, bg, N, args.data, device)
    B = dense.shape[0]
    with torch.no_grad():
        pred, _, _ = core(dense, mask, grid, bd['time_steps'], S, c)
    lp = core._last_pass; outputs, qi = lp['outputs'], lp['query_idx']
    truth = bd['data']; tm = bd['mask']; TQ = qi.shape[0]
    flat = lambda x: x.reshape(B * N, TQ, -1)
    res['base'].append(per_seq(flat(pred), truth, tm).item())
    ev = mask.any(1).any(0).nonzero().flatten()          # grid indices with an observation
    L_idx = int(ev.max())
    # ---- lagged-average forecasting
    with torch.no_grad():
        for L in tcli.lags:
            starts = [int(ev[len(ev) - 1 - k * tcli.stride]) for k in range(L) if len(ev) - 1 - k * tcli.stride >= 0]
            preds = []
            for j in starts:
                traj = rollout(outputs[:, :, j], grid, j, tau_at(mask, grid, j))
                preds.append(core.decoder(traj[:, :, qi - j]))
            res[f'lag{L}'].append(per_seq(flat(torch.stack(preds).mean(0)), truth, tm).item())
    # ---- strong-constraint 4D-Var over the last `w` of the context
    for w in tcli.windows:
        t_start = grid[L_idx] - w * (grid[L_idx] - grid[0])
        j0 = int(ev[(grid[ev] >= t_start).nonzero()[0]])
        hb = outputs[:, :, j0].detach()
        tau0 = tau_at(mask, grid, j0)
        win_mask = mask[:, :, j0:L_idx + 1].float()
        y = dense[:, :, j0:L_idx + 1]
        delta = torch.zeros_like(hb, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=tcli.lr)
        for it in range(tcli.iters):
            opt.zero_grad()
            traj = rollout(hb + delta, grid[:L_idx + 1], j0, tau0, exact=False)       # fixed-step for speed
            yhat = core.decoder(traj)
            J = (((yhat - y) ** 2).sum(-1) * win_mask).sum() / win_mask.sum().clamp(min=1) + tcli.bg * delta.pow(2).sum(-1).mean()
            J.backward(); opt.step()
        with torch.no_grad():
            traj = rollout(hb + delta, grid, j0, tau0)
            res[f'4dvar{w}'].append(per_seq(flat(core.decoder(traj[:, :, qi - j0])), truth, tm).item())
    print('batch', bi, {k: round(v[-1] * 100, 3) for k, v in res.items()}, flush=True)
out = {k: float(np.mean(v) * 100) for k, v in res.items()}
out.update(ckpt=tcli.ckpt.split('/')[-1], data=args.data, seed=args.random_seed)
print('RESULT', json.dumps(out))
if tcli.out:
    open(tcli.out, 'a').write(json.dumps(out) + '\n')

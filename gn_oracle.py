'''
Gauss-Newton launch-correction ORACLE (CHANGES.md Part 42), evaluation only, diagnostic.

Question: can a small, physical (observation-space) correction to GIL-ODE's forecast launch
state -- leaving the decoder-null memory untouched -- make the frozen model's full-horizon
forecast beat LG-ODE? The correction uses the future targets, so it is an upper bound and is never
available at inference; it tells us whether a context-only launch adapter is worth building.

For each trajectory, from the filtered state h0 at the last observation:
  F(dy) = decode(rollout(h0 + dy K^T))            dy in R^{N x D}, K the anchor gain (row space)
  J     = dF/d(dy) at dy = 0                       by forward finite differences (N*D rollouts)
  dy*   = -(J^T J + lam s I)^{-1} J^T r,  r = (F(0) - y) * mask   for a lam grid
then each agent's correction is clipped to an RMS budget, the lam with the smallest LINEARISED
residual is kept per trajectory, and the result is re-scored with a TRUE nonlinear rollout.
Standard metric (per-sequence, per-feature time average, mean), averaged over batches.
'''
import argparse, json, sys
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--split', default='val', choices=['val', 'test'])
p.add_argument('--budgets', type=float, nargs='+', default=[0.005, 0.01, 0.025, 0.05, 0.1, 1e9])
p.add_argument('--lambdas', type=float, nargs='+', default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
p.add_argument('--fd-eps', type=float, default=1e-3)
p.add_argument('--out', required=True)
ocli = p.parse_args()
sys.argv = ['x', '--ckpt', ocli.ckpt]
src = open('eval_gilode.py').read().split("N = args.n_balls")[0]
exec(src)
from lib.gil_dataset import prepare_gil_batch
assert args.mode == 'extrap'
N = args.n_balls


def per_seq(pred, truth, m):
    return torch.nanmean((((pred - truth) ** 2 * m).sum(1) / m.sum(1)).mean(-1)).item()


def step_mse(pred, truth, m):
    return (((pred - truth) ** 2 * m).sum((0, 2)) / m.sum((0, 2)).clamp(min=1)).cpu().numpy()


def tau_at(obs_mask, grid, j):
    T = grid.shape[0]
    idx = torch.arange(T, device=grid.device).view(1, 1, T)
    last = torch.where(obs_mask[:, :, :j + 1], idx[:, :, :j + 1], torch.full_like(idx[:, :, :j + 1], -1)).amax(-1)
    t_last = torch.where(last >= 0, grid[last.clamp(min=0)], grid[0] - 10.0)
    return (grid[j] - t_last).clamp(max=10.0)


def rollout(h, grid, j, tau):
    out = [h]
    for k in range(j, grid.shape[0] - 1):
        t0, t1 = grid[k], grid[k + 1]
        if (t1 - t0).abs() > 1e-8:
            core.ode_func.set_tau(tau, t0)
            h = core._integrate(h, t0, t1)
            tau = tau + (t1 - t0)
        out.append(h)
    return torch.stack(out, 2)


res = {b: {'mse': [], 'steps': [], 'pos_rms': [], 'vel_rms': [], 'lin': []} for b in ['base'] + ocli.budgets}
check = []
enc, dec, gr, nb = loaded[ocli.split]
with torch.no_grad():
    for bi in range(nb):
        be = utils.get_next_batch_new(enc, device); bg = utils.get_next_batch_new(gr, device); bd = utils.get_next_batch(dec, device)
        dense, mask, grid, S, c = prepare_gil_batch(be, bd, bg, N, args.data, device)
        B = dense.shape[0]
        pred0, _, _ = core(dense, mask, grid, bd['time_steps'], S, c)
        lp = core._last_pass; outputs, qi = lp['outputs'], lp['query_idx']
        L = int(mask.any(1).any(0).nonzero().max())
        h0 = outputs[:, :, L]; tau0 = tau_at(mask, grid, L)
        Kg = core.anchor_gain()                                  # [H, D]
        TQ = qi.shape[0]; D = dense.shape[-1]
        y = bd['data'].view(B, N, TQ, D); m = bd['mask'].view(B, N, TQ, D)

        def F(dy):
            traj = rollout(h0 + dy @ Kg.transpose(0, 1), grid, L, tau0)
            return core.decoder(traj[:, :, qi - L])

        F0 = F(torch.zeros(B, N, D, device=device))
        check.append((F0 - pred0).abs().max().item())
        P = N * D
        r = ((F0 - y) * m).reshape(B, -1)
        J = torch.zeros(B, r.shape[1], P, device=device)
        for k in range(P):
            e = torch.zeros(B, P, device=device); e[:, k] = ocli.fd_eps
            J[:, :, k] = ((F(e.view(B, N, D)) - F0) * m).reshape(B, -1) / ocli.fd_eps
        JtJ = J.transpose(1, 2) @ J
        Jtr = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)
        scale = JtJ.diagonal(dim1=1, dim2=2).mean(-1).clamp(min=1e-12).view(B, 1, 1)
        eye = torch.eye(P, device=device).unsqueeze(0)
        cands = [-torch.linalg.solve(JtJ + lam * scale * eye, Jtr.unsqueeze(-1)).squeeze(-1) for lam in ocli.lambdas]

        flat = lambda x: x.reshape(B * N, TQ, D)
        res['base']['mse'].append(per_seq(flat(F0), flat(y), flat(m)))
        res['base']['steps'].append(step_mse(flat(F0), flat(y), flat(m)))
        for bud in ocli.budgets:
            best, best_lin = None, None
            for d in cands:
                dv = d.view(B, N, D)
                rms = dv.pow(2).mean(-1, keepdim=True).sqrt()
                dv = dv * torch.clamp(bud / rms.clamp(min=1e-12), max=1.0)
                lin = (r + (J @ dv.reshape(B, P, 1)).squeeze(-1)).pow(2).sum(-1)       # [B]
                if best is None:
                    best, best_lin = dv.clone(), lin
                else:
                    take = lin < best_lin
                    best[take] = dv[take]; best_lin = torch.minimum(best_lin, lin)
            Fb = F(best)
            res[bud]['mse'].append(per_seq(flat(Fb), flat(y), flat(m)))
            res[bud]['steps'].append(step_mse(flat(Fb), flat(y), flat(m)))
            res[bud]['pos_rms'].append(best[..., :D // 2].pow(2).mean().sqrt().item())
            res[bud]['vel_rms'].append(best[..., D // 2:].pow(2).mean().sqrt().item())
            res[bud]['lin'].append((best_lin / m.reshape(B, -1).sum(-1).clamp(min=1)).mean().item())
        print('batch', bi, 'F(0)-baseline max diff %.2e' % check[-1],
              {str(k): round(np.mean(v['mse']) * 100, 3) for k, v in res.items()}, flush=True)

with open(ocli.out, 'a') as f:
    for k, v in res.items():
        f.write(json.dumps({'ckpt': ocli.ckpt.split('/')[-1], 'seed': args.random_seed, 'split': ocli.split,
                            'budget': k if k == 'base' else float(k), 'mse_full': float(np.mean(v['mse'])),
                            'mse_step': np.mean(v['steps'], 0).tolist(),
                            'pos_rms': float(np.mean(v['pos_rms'])) if v['pos_rms'] else 0.0,
                            'vel_rms': float(np.mean(v['vel_rms'])) if v['vel_rms'] else 0.0,
                            'baseline_check': float(max(check))}) + '\n')
print('done', args.random_seed)

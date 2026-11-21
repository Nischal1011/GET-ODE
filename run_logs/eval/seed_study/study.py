'''Why does springs-extrap seed 1993 (mixture config) fail? Mechanistic study for the paper (CHANGES.md Part 50).

Every checkpoint is evaluated on the SAME validation inputs (loader seeded once), first --batches batches.
Per checkpoint:
  A  forecast error by fifth of the horizon, position / velocity
  B  where the error lives: share of late error in the worst 10% of agent-trajectories; Spearman of
     per-trajectory late error with number of springs and with initial kinetic energy
  C  oscillation: dominant frequency of each forecast position channel vs truth (FFT over the horizon),
     median |log2 ratio|, and spectral amplitude ratio
  D  latent drift: norm of the decoder-null component ||h - hP|| at the start / middle / end of the
     forecast, and the fraction of forecast states beyond the guard radius
  E  vector-field decomposition at forecast states: ||f_self||, ||g * interaction||, ||a * residual||,
     interaction gate g, mixture-gate entropy, expert disagreement (std across experts / mean norm)
  F  linearised stability: eigenvalues of d f / d h at forecast states (8 trajectories x 3 times):
     spectral abscissa (max Re), fraction of eigenvalues with Re > 0.1
  G  interventions (selected checkpoints): late-horizon MSE with one component changed at a time
'''
import argparse, json, os, sys, math
import numpy as np, torch
sys.path.insert(0, os.getcwd())
import lib.utils as utils
from lib.baseline_gil_ode import GILODEBaseline
from lib.gil_dataset import prepare_gil_batch
from lib.corrected_dataLoader import CorrectedParseData

p = argparse.ArgumentParser()
p.add_argument('--ckpts', required=True, help='file with lines: tag path [interv]')
p.add_argument('--batches', type=int, default=2)
p.add_argument('--out', required=True)
p.add_argument('--channels-only', action='store_true', help='only the channel interventions, on every checkpoint')
cli = p.parse_args()
CHANNELS_ONLY = cli.channels_only
dev = torch.device('cuda:0')
jobs = [l.split() for l in open(cli.ckpts) if l.strip() and not l.startswith('#')]

def build(path):
    ck = torch.load(path, map_location=dev, weights_only=False); a = ck['args']
    m = GILODEBaseline(input_dim=4, hidden_dim=a.hidden_dim, num_atoms=a.n_balls, dataset=a.data,
                       obsrv_std=torch.Tensor([0.01]).to(dev), device=dev, mode=a.mode, mlp_width=a.mlp_width,
                       ode_substeps=a.ode_substeps, ode_tol=a.ode_tol, use_smoother=a.smoother,
                       smoother_mode=a.smoother_mode, use_reversion=getattr(a, 'reversion', False),
                       moe_k=getattr(a, 'moe', 1)).to(dev)
    m.load_state_dict(ck['state_dict'], strict=False); m.eval()
    return m, a

# data: once, seeded, identical for all checkpoints
_, a0 = build(jobs[0][1])
torch.manual_seed(1993); np.random.seed(1993)
dl = CorrectedParseData(a0.dataset, suffix=a0.suffix, mode=a0.mode, args=a0)
_tr = dl.load_data(sample_percent=a0.sample_percent_train, batch_size=a0.batch_size, data_type='train'); del _tr   # sets normalisation
enc, dec, graph, nb = dl.load_data(sample_percent=a0.sample_percent_test, batch_size=a0.batch_size, data_type='val')
batches = []
for bi in range(min(cli.batches, nb)):
    b_en = utils.get_next_batch_new(enc, dev); b_gr = utils.get_next_batch_new(graph, dev); b_de = utils.get_next_batch(dec, dev)
    batches.append((prepare_gil_batch(b_en, b_de, b_gr, a0.n_balls, a0.data, dev), b_de))
N = a0.n_balls

def run(model):
    core = model.core; P, T, M, extras = [], [], [], []
    with torch.no_grad():
        for (dense, mask, grid, S, c), b_de in batches:
            pred, _, _ = core(dense, mask, grid, b_de['time_steps'], S, c)
            B, _, TQ, D = pred.shape
            P.append(pred.cpu()); T.append(b_de['data'].view(B, N, TQ, D).cpu()); M.append(b_de['mask'].view(B, N, TQ, D).cpu())
            lp = core._last_pass
            extras.append(dict(out=lp['outputs'], qi=lp['query_idx'], grid=grid, mask=mask, S=S, c=c, tq=b_de['time_steps']))
    return torch.cat(P).numpy(), torch.cat(T).numpy(), torch.cat(M).numpy() > 0, extras

def late_mse(p, y, m):
    TQ = p.shape[2]; s = slice(int(TQ * 0.6), TQ)
    return float((((p - y) ** 2)[:, :, s][m[:, :, s]]).mean() * 100)

def forecast_stats(p, y, m, extras):
    r = {}
    TQ, D = p.shape[2], p.shape[3]
    r['fifths_pos'], r['fifths_vel'] = [], []
    for f in np.array_split(np.arange(TQ), 5):
        se = (p[:, :, f] - y[:, :, f]) ** 2; mm = m[:, :, f]
        r['fifths_pos'].append(float(se[..., :D // 2][mm[..., :D // 2]].mean() * 100))
        r['fifths_vel'].append(float(se[..., D // 2:][mm[..., D // 2:]].mean() * 100))
    r['late'] = late_mse(p, y, m)
    # B: concentration and correlates (per agent-trajectory, late 40%)
    s = slice(int(TQ * 0.6), TQ)
    se = ((p - y) ** 2 * m)[:, :, s].sum((2, 3)) / np.maximum(m[:, :, s].sum((2, 3)), 1)      # [B, N]
    flat = np.sort(se.reshape(-1))[::-1]
    r['top10_share'] = float(flat[:max(1, len(flat) // 10)].sum() / flat.sum())
    Sg = torch.cat([e['S'].cpu() for e in extras]).numpy()                                     # [B, N, N]
    deg = Sg.sum(2)                                                                             # springs per agent
    ke = (y[:, :, 0, D // 2:] ** 2).sum(-1)                                                    # initial KE proxy per agent
    from scipy.stats import spearmanr
    r['rho_err_degree'] = float(spearmanr(se.reshape(-1), deg.reshape(-1))[0])
    r['rho_err_ke'] = float(spearmanr(se.reshape(-1), ke.reshape(-1))[0])
    traj = se.mean(1); nspr = Sg.sum((1, 2)) / 2
    r['err_by_nsprings'] = {int(k): float(traj[nspr == k].mean() * 100) for k in np.unique(nspr)}
    # C: dominant frequency of position channels over the whole forecast
    pp = p[..., :D // 2] - p[..., :D // 2].mean(2, keepdims=True); yy = y[..., :D // 2] - y[..., :D // 2].mean(2, keepdims=True)
    Fp = np.abs(np.fft.rfft(pp, axis=2)); Fy = np.abs(np.fft.rfft(yy, axis=2))
    kp = Fp[:, :, 1:].argmax(2) + 1; ky = Fy[:, :, 1:].argmax(2) + 1
    r['freq_abs_log2_ratio_median'] = float(np.median(np.abs(np.log2(kp / ky))))
    r['freq_match_frac'] = float(np.mean(kp == ky))
    r['freq_pred_higher_frac'] = float(np.mean(kp > ky)); r['freq_pred_lower_frac'] = float(np.mean(kp < ky))
    r['spec_amp_ratio'] = float(Fp[:, :, 1:].sum() / Fy[:, :, 1:].sum())
    return r

def latent_stats(model, extras):
    core = model.core; f = core.ode_func
    K = core.anchor_gain(); Pr = K @ core.decoder.weight
    R = float(f.guard_R); r = {'guard_R': R}
    nn_all = []
    for e in extras:
        hq = e['out'][:, :, e['qi']]                                        # [B, N, TQ, H]
        nn_all.append((hq - hq @ Pr).norm(dim=-1).cpu())
    nn_ = torch.cat(nn_all)                                                 # [B, N, TQ]
    TQ = nn_.shape[2]
    r['null_norm_start_mid_end'] = [float(nn_[:, :, 0].mean()), float(nn_[:, :, TQ // 2].mean()), float(nn_[:, :, -1].mean())]
    r['beyond_guard_frac'] = float((nn_ > R).float().mean())
    hq_all = [e['out'][:, :, e['qi']] for e in extras]
    r['row_norm_start_end'] = [float(torch.cat([(h @ Pr)[:, :, 0].norm(dim=-1).cpu() for h in hq_all]).mean()),
                               float(torch.cat([(h @ Pr)[:, :, -1].norm(dim=-1).cpu() for h in hq_all]).mean())]
    return r, Pr

def decompose(model, extras):
    '''Vector-field terms at forecast states (all query steps of the first batch).'''
    core = model.core; f = core.ode_func; e = extras[0]
    f.set_graph(e['S'], e['c'])
    out = {'f_self': [], 'g_int': [], 'res': [], 'g': [], 'gate_H': [], 'exp_disagree': []}
    with torch.no_grad():
        hq = e['out'][:, :, e['qi']]
        B, Nn, TQ, H = hq.shape
        mask = e['mask']; grid = e['grid']
        idx = torch.arange(mask.shape[2], device=dev).view(1, 1, -1).expand_as(mask)
        last = torch.where(mask, idx, torch.zeros_like(idx)).amax(-1)
        t_last = grid[last]                                                 # [B, N]
        for k in range(0, TQ, max(1, TQ // 12)):
            h = hq[:, :, k]
            v = h[..., f.q_dim:]
            h_i = h.unsqueeze(2).expand(B, Nn, Nn, H); h_j = h.unsqueeze(1).expand(B, Nn, Nn, H)
            edge_in = torch.cat([h_i, h_j, h_j - h_i], -1)
            is_pos = (f.c > 0).unsqueeze(-1)
            msg = torch.where(is_pos, f.phi_pos(edge_in), f.phi_neg(edge_in))
            chi = f.chi(torch.cat([h_i, h_j], -1)); fa = chi - chi.transpose(1, 2)
            S_ = f.S.unsqueeze(-1); deg = f.S.sum(2, keepdim=True)
            m_sum = (S_ * msg).sum(2); m_mean = m_sum / deg.clamp(min=1); m_anti = (S_ * fa).sum(2)
            m_t = f.psi(torch.cat([m_sum, m_mean, m_anti, torch.log1p(deg)], -1))
            g = torch.sigmoid(f.gate_net(torch.cat([h, m_t, torch.log1p(deg)], -1)))
            int_in = torch.cat([h, m_t], -1)
            outs = [f.f_interaction(int_in)]
            if f.moe_k > 1:
                outs += [ex(int_in) for ex in f.f_experts]
                O = torch.stack(outs, -1)
                w = torch.softmax(f.moe_gate(h) / f.moe_temp, -1)
                inter = (O * w.unsqueeze(-2)).sum(-1)
                out['gate_H'].append(float(-(w * w.clamp_min(1e-12).log()).sum(-1).mean()))
                out['exp_disagree'].append(float(O.std(-1).norm(dim=-1).mean() / O.mean(-1).norm(dim=-1).mean()))
            else:
                inter = outs[0]
            tau = (e['tq'][k] - t_last).clamp(min=0).unsqueeze(-1)
            a = torch.sigmoid(f.residual_gate(torch.cat([h, torch.log1p(tau)], -1)))
            out['f_self'].append(float(f.f_self(h).norm(dim=-1).mean()))
            out['g_int'].append(float((g * inter).norm(dim=-1).mean()))
            out['res'].append(float((a * f.residual(h)).norm(dim=-1).mean()))
            out['g'].append(float(g.mean()))
    return {k: [round(x, 4) for x in v] for k, v in out.items() if v}

def jac_stats(model, extras, Pr):
    core = model.core; f = core.ode_func; e = extras[0]
    hq = e['out'][:, :, e['qi']]; B, Nn, TQ, H = hq.shape
    f.set_null_projector(Pr)
    res = {}
    for name, k in (('start', 0), ('mid', TQ // 2), ('end', TQ - 1)):
        absc, posf, imag = [], [], []
        for b in range(8):
            f.set_graph(e['S'][b:b + 1], e['c'][b:b + 1])
            f.set_tau(torch.zeros(1, Nn, device=dev), torch.tensor(0.0, device=dev))
            h0 = hq[b:b + 1, :, k].reshape(-1).detach()
            J = torch.autograd.functional.jacobian(lambda x: f(torch.tensor(0.0, device=dev), x.view(1, Nn, H)).reshape(-1), h0, vectorize=True)
            ev = torch.linalg.eigvals(J.double()).cpu().numpy()
            absc.append(ev.real.max()); posf.append(np.mean(ev.real > 0.1)); imag.append(np.abs(ev.imag).max())
        res[name] = dict(abscissa=float(np.mean(absc)), frac_re_gt_0p1=float(np.mean(posf)), max_imag=float(np.mean(imag)))
    f._tau_base = None
    return res

def interventions(model, base_late):
    core = model.core; f = core.ode_func; r = {'baseline': base_late}
    def measure(): p, y, m, _ = run(model); return late_mse(p, y, m)
    if f.moe_k > 1 and not CHANNELS_ONLY:
        orig = f.moe_gate.forward
        for k in range(f.moe_k):
            def one_hot(x, k=k):
                o = torch.full(x.shape[:-1] + (f.moe_k,), -1e4, device=x.device); o[..., k] = 0.0; return o
            f.moe_gate.forward = one_hot; r[f'expert_{k}_only'] = measure()
        f.moe_gate.forward = lambda x: torch.zeros(x.shape[:-1] + (f.moe_k,), device=x.device); r['uniform_mixture'] = measure()
        f.moe_gate.forward = orig
    if core.use_reversion and not CHANNELS_ONLY:
        core.use_reversion = False; r['no_reversion'] = measure(); core.use_reversion = True
    for ab in (() if CHANNELS_ONLY else ('no_residual', 'no_guard', 'no_antisym')):
        f.ablate.add(ab); r[ab] = measure(); f.ablate.discard(ab)
    if CHANNELS_ONLY:
        r = {'baseline': base_late}
    for ab in (('no_antisym',) if CHANNELS_ONLY else ()):
        f.ablate.add(ab); r[ab] = measure(); f.ablate.discard(ab)
    o_pos, o_neg = f.phi_pos.forward, f.phi_neg.forward      # free-form (non-antisymmetric) messages off
    f.phi_pos.forward = lambda x: torch.zeros(x.shape[:-1] + (f.phi_pos[-1].out_features,), device=x.device)
    f.phi_neg.forward = lambda x: torch.zeros(x.shape[:-1] + (f.phi_neg[-1].out_features,), device=x.device)
    r['no_free_messages'] = measure()
    f.phi_pos.forward, f.phi_neg.forward = o_pos, o_neg
    if CHANNELS_ONLY:
        return r
    og = f.gate_net.forward
    for s in (0.5, 0.0):
        def scaled(x, s=s):      # logit whose sigmoid is s * sigmoid(original)
            if s == 0: return torch.full_like(og(x), -1e4)
            g = (torch.sigmoid(og(x)) * s).clamp(1e-6, 1 - 1e-6); return torch.log(g / (1 - g))
        f.gate_net.forward = scaled
        r[f'interaction_gate_x{s}' if s > 0 else 'interaction_off'] = measure()
    f.gate_net.forward = og
    return r

results = {}
for job in jobs:
    tag, path = job[0], job[1]; do_int = len(job) > 2 and job[2] == 'interv'
    model, a = build(path)
    p, y, m, extras = run(model)
    r = {'ckpt': os.path.basename(path), 'seed': a.random_seed, 'moe': getattr(a, 'moe', 1)}
    r.update(forecast_stats(p, y, m, extras))
    ls, Pr = latent_stats(model, extras); r.update(ls)
    if not CHANNELS_ONLY:
        r['decomp'] = decompose(model, extras)
        r['jac'] = jac_stats(model, extras, Pr)
    if do_int:
        r['interv'] = interventions(model, r['late'])
    results[tag] = r
    print(tag, json.dumps({k: v for k, v in r.items() if k not in ('decomp',)}), flush=True)
    del model, extras; torch.cuda.empty_cache()
json.dump(results, open(cli.out, 'w'), indent=1)
print('STUDYDONE')

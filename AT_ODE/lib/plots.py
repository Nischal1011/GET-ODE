from __future__ import annotations
import os, numpy as np, torch
from typing import Iterable, List, Tuple, Optional, Dict, Union
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


Array = Union[np.ndarray, "torch.Tensor"]  # type: ignore

# ---------- helpers ----------
def _to_numpy(x: Array) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _resolve_path(func_name: str, savepath: Optional[str], save_dir: Optional[str]) -> str:
    if savepath is not None:
        out = savepath
    else:
        folder = "." if (save_dir is None) else save_dir
        out = os.path.join(folder, f"{func_name}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    return out

# --- helpers inside run_models_gpu.py (near do_eval_and_plots or above it) ---
def _denorm_times_for_display(t_norm: torch.Tensor, args, phase="test"):
    """
    Reverse the normalization used in ParseData.interp_extrap.
    interp:   t_real = t_norm * total_step
    extrap:   test: t_real = t_norm * total_step + total_step
              raw: t_real = t_norm * total_step + total_step/2
    """
    S = float(getattr(args, "total_ode_step", 1))
    t = t_norm.detach()
    if getattr(args, "mode", "interp") == "interp":
        return t * S
    else:
        offset = S if phase == "test" else (S / 2.0)
        return t * S + offset

def _align_encoder_mask_to_decoder_grid(batch_enc, batch_dec, feat_dim: int) -> torch.Tensor:
    """
    Returns a boolean mask [M_dec, T_dec] telling which nodes are observed at each decoder time
    according to the *encoder* mask/times. If encoder keys are missing, returns None.
    """
    if ("time_steps" not in batch_enc) or ("mask" not in batch_enc):
        return None
    enc_tt = batch_enc["time_steps"]          # [T_enc]
    enc_mk = batch_enc["mask"]                # [M_enc, T_enc, D_enc]
    dec_tt = batch_dec["time_steps"]          # [T_dec]

    mk_any = (enc_mk[..., :feat_dim] > 0.5).any(dim=-1)   # [M_enc, T_enc]
    match = torch.isclose(enc_tt.view(-1,1), dec_tt.view(1,-1), atol=1e-6, rtol=1e-5)  # [T_enc, T_dec]
    obs_on_dec = (mk_any.unsqueeze(-1) & match.unsqueeze(0)).any(dim=1)                # [M_enc, T_dec]
    return obs_on_dec

# ---------- UQ utilities ----------
def series_mean_std(pred_samples: Array, obsrv_std: float | Array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ps = _to_numpy(pred_samples)
    mu = ps.mean(axis=0)
    var_ep = ps.var(axis=0, ddof=1)
    sig_ep = np.sqrt(np.maximum(var_ep, 0.0))
    sig_tot = np.sqrt(sig_ep**2) # + _to_numpy(obsrv_std)**2)
    return mu, sig_ep, sig_tot

def plot_series_with_ci(
    tt,                # unused (kept for signature compatibility)
    truth,             # [T, D] decoder truth on the same grid as mu/sigma/mask
    mu,                # [T, D] decoder mean
    sigma,             # [T, D] decoder std (total)
    mask,              # [T, D] decoder mask (1 if observed at that time)
    feat_idx: int,
    title: str,
    savepath: str,
    type_name: str | None,
    t_display,         # [T] decoder time axis (already de-normalized or normalized – your choice)
    feature_name: str | None = None,   # <<< NEW (optional)
    bus_id: str | None = None,         # <<< NEW (optional)
    node_idx: int | None = None,       # <<< NEW (optional)
    **_ignored                           # swallow legacy obs_* kwargs cleanly
) -> str:
    import numpy as np, matplotlib.pyplot as plt, os

    # convert to numpy
    def _np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    t   = _np(t_display).reshape(-1)
    Y   = _np(truth)[:, feat_idx]
    MU  = _np(mu)[:, feat_idx]
    SIG = _np(sigma)[:, feat_idx]
    MK  = (_np(mask)[:, feat_idx] > 0.5)

    # observed decoder times/values
    t_obs  = t[MK]
    y_obs  = Y[MK]
    mu_obs = MU[MK]

    lo = MU - 1.96 * SIG
    hi = MU + 1.96 * SIG

    # Compose a clear title if caller didn't build one
    if not title:
        bits = []
        if type_name: bits.append(f"[{type_name}]")
        if node_idx is not None: bits.append(f"Node {node_idx}")
        if bus_id is not None: bits.append(f"(bus {bus_id})")
        if feature_name: bits.append(feature_name)
        bits.append("±95% CI")
        title = " ".join(bits)

    ylabel = feature_name or f"feature {feat_idx}"

    plt.figure(figsize=(8, 3.2), dpi=160)

    # Uncertainty band + mean curve
    plt.fill_between(t, lo, hi, alpha=0.22, linewidth=0, label="95% CI")
    plt.plot(t, MU, "--", linewidth=2.0, label="pred mean")

    # Observations (orange dots) and predictions at those exact times (blue ×)
    if t_obs.size:
        plt.scatter(t_obs, y_obs, s=10, label="truth (obs)", color="#FF7F0E")
        plt.scatter(t_obs, mu_obs, s=36, marker="x", linewidths=1.5, label="pred @ obs-time", color="#1f77b4")
        # optional: crosses at every decoder time to show imputation path
        plt.scatter(t, MU, s=24, marker="x", linewidths=0.9, alpha=0.25, label="pred @ dec-times", color="#1f77b4")

    plt.title(title)
    plt.xlabel("time"); plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight"); plt.close()
    return savepath

# ---------- node metrics ----------
def per_node_rmse(truths: List[Array], mus: List[Array], masks: Optional[List[Array]] = None) -> np.ndarray:
    rmses = []
    for i, (y, m) in enumerate(zip(truths, mus)):
        y, m = _to_numpy(y), _to_numpy(m)
        if masks is not None:
            w = _to_numpy(masks[i]).astype(bool)
        else:
            w = np.isfinite(y) & np.isfinite(m)
        if not np.any(w):
            rmses.append(np.nan)
        else:
            e2 = (y[w] - m[w]) ** 2
            rmses.append(np.sqrt(e2.mean()))
    return np.array(rmses)

def plot_node_bars(values: np.ndarray, title: str,
                   xlabel: str = "nodes (sorted)", ylabel: str = "RMSE",
                   savepath: Optional[str] = None, save_dir: Optional[str] = None) -> str:
    order = np.argsort(np.nan_to_num(values, nan=np.inf))
    plt.figure(figsize=(8,3), dpi=160)
    plt.bar(np.arange(len(values)), values[order])
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    out = _resolve_path("plot_node_bars", savepath, save_dir)
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close()
    return out

# ---------- calibration ----------
def coverage_curve(y: Array, mu: Array, sigma: Array,
                   levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
                   mask: Optional[Array] = None) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.stats import norm
    yy, mm, ss = _to_numpy(y), _to_numpy(mu), _to_numpy(sigma)
    if mask is not None:
        mk = _to_numpy(mask).astype(bool)
        yy, mm, ss = yy[mk], mm[mk], ss[mk]
    else:
        yy, mm, ss = yy.ravel(), mm.ravel(), ss.ravel()
    lev = np.array(list(levels), dtype=float)
    cov = []
    for q in lev:
        z = norm.ppf((1 + q) / 2.0)
        inside = (yy >= mm - z * ss) & (yy <= mm + z * ss)
        cov.append(inside.mean() if yy.size else np.nan)
    return lev, np.array(cov)

def plot_coverage(levels: Array, coverage: Array,
                  savepath: Optional[str] = None, save_dir: Optional[str] = None) -> str:
    x, y = _to_numpy(levels), _to_numpy(coverage)
    plt.figure(figsize=(4.5, 3.5), dpi=170)
    plt.plot(x, y, marker="o", linewidth=1.8, label="empirical")
    plt.plot([0,1],[0,1], "--", linewidth=1.2, alpha=0.6, label="ideal")
    plt.xlabel("nominal CI level"); plt.ylabel("empirical coverage")
    plt.title("Uncertainty Calibration")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out = _resolve_path("plot_coverage", savepath, save_dir)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def plot_uncertainty_vs_error(truth: Array, mu: Array, sigma: Array, mask: Optional[Array] = None,
                              savepath: Optional[Array] = None, save_dir: Optional[str] = None) -> str:
    y = _to_numpy(truth); m = _to_numpy(mu); s = _to_numpy(sigma)
    if mask is not None:
        mk = _to_numpy(mask).astype(bool)
        y, m, s = y[mk], m[mk], s[mk]
    err = np.abs(y - m).ravel(); sig = s.ravel()
    plt.figure(figsize=(4.5, 3.5), dpi=170)
    plt.scatter(sig, err, s=8, alpha=0.25)
    plt.xlabel("predicted σ"); plt.ylabel("|error|")
    plt.title("Uncertainty vs Error")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    out = _resolve_path("uncertainty_vs_error", savepath, save_dir)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

# ---------- attention aggregation & graph viz ----------
def type_attention_matrix(alpha: Array,
                          t_send: Array,
                          t_recv: Array,
                          n_types: int) -> Tuple[np.ndarray, np.ndarray]:
    a = _to_numpy(alpha).reshape(-1)
    ts = _to_numpy(t_send).astype(int).reshape(-1)
    tr = _to_numpy(t_recv).astype(int).reshape(-1)
    M = np.zeros((n_types, n_types), dtype=float)
    C = np.zeros((n_types, n_types), dtype=int)
    for ai, s, r in zip(a, ts, tr):
        if 0 <= s < n_types and 0 <= r < n_types:
            M[r, s] += float(ai)
            C[r, s] += 1
    rowsum = M.sum(axis=1, keepdims=True)
    nz = rowsum.squeeze() > 0
    M[nz] = M[nz] / rowsum[nz]
    return M, C

def graph_attention_edges(edge_index: Array,
                          alpha: Array,
                          num_nodes: Optional[int] = None,
                          node_metric: Optional[Array] = None,
                          sensor_idx: Optional[Array] = None,
                          type_names: Optional[List[str]] = None,
                          pos_xy: Optional[Array] = None,
                          topk: int = 300,
                          threshold: Optional[float] = None,
                          undirected: bool = False,
                          width_scale: float = 6.0,
                          title: str = "Attention graph",
                          savepath: Optional[str] = None,
                          save_dir: Optional[str] = None) -> str:
    ei = _to_numpy(edge_index)
    a  = _to_numpy(alpha).reshape(-1)
    E = ei.shape[1]
    N = int(num_nodes) if num_nodes is not None else int(ei.max()) + 1

    order = np.argsort(-a)
    sel = order[:min(topk, E)] if threshold is None else np.where(a >= float(threshold))[0]
    sel = np.asarray(sel, dtype=int)

    if pos_xy is None:
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        pos = np.stack([np.cos(t), np.sin(t)], axis=1)
    else:
        pos = _to_numpy(pos_xy)
        if pos.shape[0] != N:
            raise ValueError("pos_xy has wrong number of nodes")

    if sensor_idx is not None:
        st = _to_numpy(sensor_idx).astype(int)
        uniq = np.unique(st)
        cmap = plt.get_cmap("tab10")
        colors = {u: cmap(int(i % 10)) for i, u in enumerate(uniq)}
        node_colors = np.array([colors.get(s, (0.5,0.5,0.5,1.0)) for s in st])
    else:
        node_colors = np.full((N,4), (0.4,0.4,0.4,1.0))

    if node_metric is not None:
        nm = _to_numpy(node_metric).astype(float)
        nm = (nm - np.nanmin(nm)) / (np.nanmax(nm) - np.nanmin(nm) + 1e-12)
        sizes = 100 * (0.6 + 0.8 * (1 - nm))
    else:
        sizes = np.full(N, 120.0)

    plt.figure(figsize=(6,6), dpi=170)
    a_sel = a[sel]
    widths = (width_scale * (a_sel / a_sel.max())) if (a_sel.max() > 0) else np.full_like(a_sel, 0.1)
    for k, e in enumerate(sel):
        i, j = int(ei[0, e]), int(ei[1, e])
        plt.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], linewidth=float(widths[k]), alpha=0.35, color="k")
    plt.scatter(pos[:,0], pos[:,1], s=sizes, c=node_colors, edgecolor="k", linewidths=0.4, zorder=3)
    plt.axis("off"); plt.title(title); plt.tight_layout()
    out = _resolve_path("graph_attention_edges", savepath, save_dir)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

# ---------- three clean grid plots + animator ----------

def plot_grid_observed(edge_index, XY, types, type_names, obs_mask_t, title, savepath, node_size=20.0):
    ei = _to_numpy(edge_index); N = XY.shape[0]
    ty = _to_numpy(types).astype(int) if types is not None else np.zeros(N, int)
    palette = plt.cm.tab10(np.linspace(0,1,max(10, len(type_names or []))))[:max(1, len(type_names or [0]))]
    base = palette[ty % len(palette)]

    plt.figure(figsize=(6.2,6), dpi=180)
    for u,v in ei.T:
        plt.plot([XY[u,0], XY[v,0]], [XY[u,1], XY[v,1]], color='k', linewidth=0.7, alpha=0.35, zorder=1)
    obs = _to_numpy(obs_mask_t).astype(bool).reshape(-1)
    idx_un = np.where(~obs)[0]
    if idx_un.size:
        plt.scatter(XY[idx_un,0], XY[idx_un,1], s=node_size, facecolors="none",
                    edgecolors=base[idx_un], linewidths=0.8, alpha=0.25, zorder=2)
    idx_ob = np.where(obs)[0]
    if idx_ob.size:
        plt.scatter(XY[idx_ob,0], XY[idx_ob,1], s=node_size*1.05, facecolors=base[idx_ob],
                    edgecolors='k', linewidths=0.6, alpha=0.95, zorder=3)
    plt.title(title); plt.axis("off"); plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    out = _resolve_path("grid_observed", savepath, None)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def plot_grid_predicted(edge_index, XY, pred_values_t, title, savepath, node_size=20.0):
    ei = _to_numpy(edge_index); N = XY.shape[0]
    v = _to_numpy(pred_values_t).reshape(-1)
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    plt.figure(figsize=(6.2,6), dpi=180)
    for u,vv in ei.T:
        plt.plot([XY[u,0], XY[vv,0]], [XY[u,1], XY[vv,1]], color='k', linewidth=0.7, alpha=0.35, zorder=1)
    sc = plt.scatter(XY[:,0], XY[:,1], s=node_size, c=v, cmap="inferno",
                     vmin=vmin, vmax=vmax, edgecolors='k', linewidths=0.6, zorder=3)
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.03); cbar.set_label("predicted value")
    plt.title(title); plt.axis("off"); plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    out = _resolve_path("grid_predicted", savepath, None)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def plot_grid_error(edge_index, XY, err_t, types, type_names, title, savepath, node_size=20.0):
    ei = _to_numpy(edge_index); N = XY.shape[0]
    ty = _to_numpy(types).astype(int) if types is not None else np.zeros(N, int)
    palette = plt.cm.tab10(np.linspace(0,1,max(10, len(type_names or []))))[:max(1, len(type_names or [0]))]
    base = palette[ty % len(palette)]
    e = _to_numpy(err_t).reshape(-1)

    # percentile range for visual contrast
    e_obs = e[np.isfinite(e)]
    if e_obs.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanpercentile(e_obs, 5)), float(np.nanpercentile(e_obs, 95))
        vmax = max(vmax, vmin + 1e-6)

    en = np.clip((e - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = base[:,:3]
    light = rgb + (1 - rgb) * 0.60
    dark  = rgb * 0.35
    shade = (1 - en[:,None]) * light + en[:,None] * dark
    shade[~np.isfinite(e), :] = 1.0  # white for unobserved/invalid

    plt.figure(figsize=(6.2,6), dpi=180)
    for u,v in ei.T:
        plt.plot([XY[u,0], XY[v,0]], [XY[u,1], XY[v,1]], color='k', linewidth=0.7, alpha=0.35, zorder=1)
    sc = plt.scatter(XY[:,0], XY[:,1], s=node_size, c=np.ma.masked_invalid(e), cmap="magma",
                     edgecolors='k', linewidths=0.4, zorder=2)
    plt.scatter(XY[:,0], XY[:,1], s=node_size*0.95, facecolors=np.c_[shade, np.ones((N,1))],
                edgecolors='none', zorder=3)
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.03); cbar.set_label("abs. error at t")
    plt.title(title); plt.axis("off"); plt.gca().set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    out = _resolve_path("grid_error", savepath, None)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def animate_network_three(edge_index, t_seconds, truth, mu, mask,
                          types, type_names, feat_idx,
                          pos_xy, save_prefix, interval_ms=800, step=1,
                          obs_mask_override=None, feature_name=None):
    """
    Writes three GIFs:
      <save_prefix>_observed.gif, <save_prefix>_predicted.gif, <save_prefix>_error.gif
    All animations are heatmaps for the selected feat_idx.
    Nodes with NO observation at a time are drawn BLACK (observed/predicted/error).
    """


    # ---------- helpers ----------
    def _np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _edges(ax, XY, ei):
        for u, v in ei.T:
            ax.plot([XY[u, 0], XY[v, 0]], [XY[u, 1], XY[v, 1]],
                    color='k', linewidth=0.7, alpha=0.35, zorder=1)

    # ---------- to numpy ----------
    ei = _np(edge_index)
    Y  = _np(truth)    # [N, T, D]
    M  = _np(mu)       # [N, T, D]
    Mk = _np(mask).astype(bool)
    t  = _np(t_seconds).reshape(-1)

    N, T, D = Y.shape
    feat_label = str(feature_name) if feature_name else f"feature {feat_idx}"
    feat_idx = int(feat_idx)

    # ---------- layout ----------
    if pos_xy is not None and _np(pos_xy).shape == (N, 2):
        XY = _np(pos_xy)
    else:
        try:
            import networkx as nx
            G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(ei.T.tolist())
            XYd = nx.spring_layout(G, seed=0, dim=2)
            XY = np.vstack([XYd[i] for i in range(N)])
        except Exception:
            ang = np.linspace(0, 2*np.pi, N, endpoint=False)
            XY = np.c_[np.cos(ang), np.sin(ang)]

    # ---------- frames & masks (only this feature) ----------
    frames = list(range(0, T, max(1, int(step))))
    if obs_mask_override is not None:
        obs_mask_full = _np(obs_mask_override).astype(bool)  # [N, T]
    else:
        obs_mask_full = Mk[:, :, feat_idx]                    # [N, T]

    # ---------- values for this feature ----------
    V_true = Y[:, :, feat_idx]   # [N, T]
    V_pred = M[:, :, feat_idx]   # [N, T]

    # shared scale (use ONLY observed positions)
    TrueObs = np.where(obs_mask_full, V_true, np.nan)
    PredObs = np.where(obs_mask_full, V_pred, np.nan)
    if np.isfinite(TrueObs).any() or np.isfinite(PredObs).any():
        vmin = np.nanmin([np.nanmin(TrueObs), np.nanmin(PredObs)])
        vmax = np.nanmax([np.nanmax(TrueObs), np.nanmax(PredObs)])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        # No observations anywhere: we still set a benign scale,
        # but every node will remain black in all three animations.
        vmin, vmax = 0.0, 1.0

    # error scale from observed errors
    abs_err_all = np.abs(V_pred - V_true)                 # [N, T]
    obs_err = np.where(obs_mask_full, abs_err_all, np.nan)
    err_max = float(np.nanpercentile(obs_err, 95)) if np.isfinite(obs_err).any() else 1.0
    err_max = max(err_max, 1e-12)

    # ===================== 1) OBSERVED HEATMAP =====================
    fig1, ax1 = plt.subplots(figsize=(6.2, 6), dpi=170)
    _edges(ax1, XY, ei)
    # base layer: ALL nodes black
    bg1 = ax1.scatter(XY[:, 0], XY[:, 1], s=22, c='k', edgecolors='k', linewidths=0.4, zorder=2)
    # overlay ONLY observed nodes
    obs0  = obs_mask_full[:, 0]
    sc1 = ax1.scatter(XY[obs0, 0], XY[obs0, 1], s=22, c=V_true[obs0, 0],
                      cmap="inferno", vmin=vmin, vmax=vmax,
                      edgecolors='k', linewidths=0.6, zorder=3)
    cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.03); cbar1.set_label(feat_label)
    ts_txt1 = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, ha="left", va="top",
                       bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"))
    ax1.axis("off"); ax1.set_title("Observed heatmap")
    ax1.legend(handles=[Line2D([], [], marker='s', linestyle='', markersize=8,
                               markerfacecolor='none', markeredgecolor='k',
                               label=feat_label)],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))

    def upd_obs(k):
        tidx = frames[k]
        obs = obs_mask_full[:, tidx]
        sc1.set_offsets(XY[obs])
        sc1.set_array(V_true[obs, tidx])
        ts_txt1.set_text(f"t ≈ {float(t[tidx]):.4g}")
        return sc1, ts_txt1

    ani1 = animation.FuncAnimation(fig1, upd_obs, frames=len(frames),
                                   interval=interval_ms, blit=False)
    try:
        ani1.save(f"{save_prefix}_observed.gif", writer="pillow", dpi=160)
    except Exception:
        pass
    plt.close(fig1)

    # ===================== 2) PREDICTED HEATMAP =====================
    fig2, ax2 = plt.subplots(figsize=(6.2, 6), dpi=170)
    _edges(ax2, XY, ei)
    # base layer black
    bg2 = ax2.scatter(XY[:, 0], XY[:, 1], s=22, c='k', edgecolors='k', linewidths=0.4, zorder=2)
    # overlay ONLY where observed at this time
    obs0 = obs_mask_full[:, 0]
    sc2 = ax2.scatter(XY[obs0, 0], XY[obs0, 1], s=22, c=V_pred[obs0, 0],
                      cmap="inferno", vmin=vmin, vmax=vmax,
                      edgecolors='k', linewidths=0.6, zorder=3)
    cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.03); cbar2.set_label(feat_label)
    ts_txt2 = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, ha="left", va="top",
                       bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"))
    ax2.axis("off"); ax2.set_title("Predicted heatmap")
    ax2.legend(handles=[Line2D([], [], marker='s', linestyle='', markersize=8,
                               markerfacecolor='none', markeredgecolor='k',
                               label=feat_label)],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))

    def upd_pred(k):
        tidx = frames[k]
        obs = obs_mask_full[:, tidx]
        sc2.set_offsets(XY[obs])
        sc2.set_array(V_pred[obs, tidx])
        ts_txt2.set_text(f"t ≈ {float(t[tidx]):.4g}")
        return sc2, ts_txt2

    ani2 = animation.FuncAnimation(fig2, upd_pred, frames=len(frames),
                                   interval=interval_ms, blit=False)
    try:
        ani2.save(f"{save_prefix}_predicted.gif", writer="pillow", dpi=160)
    except Exception:
        pass
    plt.close(fig2)

    # ===================== 3) ERROR HEATMAP =====================
    fig3, ax3 = plt.subplots(figsize=(6.2, 6), dpi=170)
    _edges(ax3, XY, ei)
    # base layer black
    bg3 = ax3.scatter(XY[:, 0], XY[:, 1], s=22, c='k', edgecolors='k', linewidths=0.4, zorder=2)
    # overlay ONLY observed errors
    e0 = abs_err_all[:, 0]
    obs0 = obs_mask_full[:, 0]
    sc3 = ax3.scatter(XY[obs0, 0], XY[obs0, 1], s=22, c=e0[obs0], cmap="magma",
                      vmin=0.0, vmax=err_max, edgecolors='k', linewidths=0.4, zorder=3)
    cbar3 = plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.03)
    cbar3.set_label(f"abs. error • {feat_label}")
    ts_txt3 = ax3.text(0.02, 0.98, "", transform=ax3.transAxes, ha="left", va="top",
                       bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"))
    ax3.axis("off"); ax3.set_title("Prediction error heatmap")
    ax3.legend(handles=[Line2D([], [], marker='s', linestyle='', markersize=8,
                               markerfacecolor='none', markeredgecolor='k',
                               label=feat_label)],
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))

    def upd_err(k):
        tidx = frames[k]
        obs = obs_mask_full[:, tidx]
        sc3.set_offsets(XY[obs])
        sc3.set_array(abs_err_all[obs, tidx])
        ts_txt3.set_text(f"t ≈ {float(t[tidx]):.4g}")
        return sc3, ts_txt3

    ani3 = animation.FuncAnimation(fig3, upd_err, frames=len(frames),
                                   interval=interval_ms, blit=False)
    try:
        ani3.save(f"{save_prefix}_error.gif", writer="pillow", dpi=160)
    except Exception:
        pass
    plt.close(fig3)

    return f"{save_prefix}_observed.gif", f"{save_prefix}_predicted.gif", f"{save_prefix}_error.gif"

def plot_training_curves(train: Dict[str, List[float]],
                         test: Dict[str, List[float]],
                         savepath: str) -> str:
    """
    Plot epoch-wise training/test curves for matching metric keys.
    Only draws the test curve for a key if its length equals the raw length.
    Returns the saved file path.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # choose keys: prefer common ones, then any extras in stable order
    preferred = ["loss", "mse", "lik"]
    keys = [k for k in preferred if (k in train) or (k in test)]
    extras = [k for k in list(train.keys()) + list(test.keys()) if k not in keys]
    # keep first occurrence order for extras
    seen = set(keys)
    for k in extras:
        if k not in seen:
            keys.append(k); seen.add(k)

    if not keys:
        raise ValueError("plot_training_curves: no metric keys found in raw/test dicts.")

    ncols = len(keys)
    plt.figure(figsize=(3.6 * ncols, 3.2), dpi=160)

    for i, key in enumerate(keys, start=1):
        tr = train.get(key, [])
        te = test.get(key, [])
        epochs = np.arange(1, len(tr) + 1)

        ax = plt.subplot(1, ncols, i)
        if len(tr) > 0:
            ax.plot(epochs, tr, label="raw", linewidth=1.8)
        if len(te) == len(tr) and len(te) > 0:
            ax.plot(epochs, te, "--", label="test", linewidth=1.4)

        ax.set_title(key.upper())
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.grid(alpha=0.25, linewidth=0.6)
        if i == 1:
            ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    # save using the same resolver style as other plots
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight"); plt.close()
    return savepath

def plot_heatmap(M: np.ndarray,
                 row_labels: list[str],
                 col_labels: list[str] | None = None,
                 annotate_counts: np.ndarray | None = None,
                 title: str = "Type↔Type Attention",
                 savepath: str | None = None,
                 save_dir: str | None = None) -> str:
    M = _to_numpy(M)
    col_labels = col_labels or row_labels
    plt.figure(figsize=(5.0, 4.0), dpi=180)
    im = plt.imshow(M, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.xlabel("sender type"); plt.ylabel("receiver type")
    if annotate_counts is not None:
        C = _to_numpy(annotate_counts)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                txt = f"{int(C[i, j])}"
                color = "w" if M[i, j] > (M.max() * 0.6) else "k"
                plt.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)
    plt.tight_layout()
    out = _resolve_path("heatmap", savepath, save_dir)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out

def _make_type_palette(type_names: list[str]) -> np.ndarray:
    import matplotlib.pyplot as plt, numpy as np
    cmap = plt.get_cmap("tab10")
    return np.array([cmap(i % 10) for i in range(len(type_names))])

def plot_structural_with_ids(edge_index, num_nodes: int,
                             sensor_idx, type_names: list[str],
                             savepath: str):
    import numpy as np, matplotlib.pyplot as plt
    # to numpy
    ei = _to_numpy(edge_index)
    st = _to_numpy(sensor_idx).astype(int)

    # layout from edges (spring), fallback circular
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(ei.T.tolist())
        posd = nx.spring_layout(G, seed=0, dim=2)
        pos = np.vstack([posd[i] for i in range(num_nodes)])
    except Exception:
        ang = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
        pos = np.c_[np.cos(ang), np.sin(ang)]

    # colors per type
    palette = _make_type_palette(type_names)
    node_colors = palette[st % len(palette)]

    plt.figure(figsize=(7.5, 7.0), dpi=170)
    # edges
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        plt.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color="k", linewidth=0.6, alpha=0.5, zorder=1)

    # nodes
    plt.scatter(pos[:,0], pos[:,1], s=42, c=node_colors, edgecolor="k", linewidths=0.4, zorder=3)

    # id labels (small)
    for i, (x, y) in enumerate(pos):
        plt.text(x, y, str(i), fontsize=6.5, ha="center", va="center", color="k", zorder=4)

    # legend with type swatches
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', markersize=6,
                          markerfacecolor=palette[t], markeredgecolor='k', label=type_names[t])
               for t in range(len(type_names))]
    plt.legend(handles=handles, frameon=False, fontsize=8, loc="lower left", ncol=2)

    plt.axis("off"); plt.title("Structural grid with node IDs"); plt.tight_layout()
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight"); plt.close()
    return savepath

def export_node_types_csv(sensor_idx, type_names: list[str], csv_path: str):
    import csv, numpy as np, os
    st = _to_numpy(sensor_idx).astype(int)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "type"])
        for i, t in enumerate(st):
            name = type_names[int(t)] if 0 <= int(t) < len(type_names) else str(t)
            w.writerow([i, name])
    return csv_path

# --- NEW: feature-wise MSE + bar plot ----------------------------------------
def mse_per_feature(truth: Array, mu: Array, mask: Array | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mse   : [D] MSE per feature, computed over positions where mask[..., d] == 1 (if provided)
      count : [D] number of points used per feature
    """
    y = _to_numpy(truth)
    m = _to_numpy(mu)
    assert y.shape == m.shape and y.ndim == 3, "truth/mu must be [M,T,D]"
    M, T, D = y.shape

    y2 = y.reshape(-1, D)
    m2 = m.reshape(-1, D)

    if mask is not None:
        mk = _to_numpy(mask).astype(bool).reshape(-1, D)
    else:
        mk = np.isfinite(y2) & np.isfinite(m2)

    mse = np.full(D, np.nan, dtype=float)
    cnt = np.zeros(D, dtype=int)
    for d in range(D):
        sel = mk[:, d]
        cnt[d] = int(sel.sum())
        if cnt[d] > 0:
            e = y2[sel, d] - m2[sel, d]
            mse[d] = float(np.mean(e * e))
    return mse, cnt


def plot_feature_bars(values: np.ndarray,
                      labels: list[str] | None = None,
                      title: str = "Feature-wise MSE",
                      xlabel: str = "features (sorted)",
                      ylabel: str = "MSE",
                      savepath: str | None = None,
                      save_dir: str | None = None) -> str:
    v = _to_numpy(values).astype(float)
    order = np.argsort(np.nan_to_num(v, nan=np.inf))
    x = np.arange(len(v))

    plt.figure(figsize=(max(6.0, 0.35 * len(v)), 3.2), dpi=160)
    plt.bar(x, v[order])
    if labels:
        labs_sorted = [labels[i] for i in order]
        plt.xticks(x, labs_sorted, rotation=35, ha="right", fontsize=8)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    out = _resolve_path("feature_mse_bars", savepath, save_dir)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    return out


# ---- put this somewhere importable (e.g., debug_snapshots.py) ----
import os, numpy as np, torch, matplotlib.pyplot as plt

@torch.no_grad()
def plot_quick_stage(model,
                     batch_enc, batch_dec, batch_graph,
                     node_idx: int = 0,      # which series (M axis)
                     feat_idx: int = 0,      # which decoded feature
                     latent_dim_to_plot: int = 0,
                     sample_id: int = 0,
                     total_ode_step: float = 1.0,
                     mode: str = "interp",
                     tag: str = "raw",
                     epoch: int | None = None,
                     step: int | None = None,
                     outdir: str = "train_snaps"):
    """
    Makes a 3-panel figure:
      (1) pred_x (node_idx, feat_idx) vs truth with observed points highlighted
      (2) latent z(t) for chosen latent dim
      (3) z0 mean ± (optional) sample marker for that node

    Assumptions: batch_size == 1 so node_idx indexes the M axis directly.
    """

    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    # ---------- run one forward to get intermediates ----------
    model.eval()
    pred_x, info, _ = model.get_reconstruction(
        batch_enc, batch_dec, batch_graph, n_traj_samples=max(1, sample_id + 1)
    )
    # pred_x: [S, M, T, D]; info["latent_traj"]:[S, M, T, Z]; info["first_point"]=(μ[1,M,Z], σ[1,M,Z], z0_samples[S,M,Z])

    z0_mu, z0_std, z0_samples = info["first_point"]
    z_traj = info["latent_traj"]

    # ---------- select series / time / feature ----------
    # time (denormalize to “real” units like you’ve been plotting)
    tt = batch_dec["time_steps"]
    if mode == "interp":
        t_real = tt * float(total_ode_step)
    else:
        offset = float(total_ode_step)  # test offset; use total_ode_step/2.0 if you're in raw-extrap
        t_real = tt * float(total_ode_step) + offset
    t_real = _np(t_real)

    MU = pred_x[sample_id, node_idx]              # [T, D]
    Y  = batch_dec["raw"][node_idx]              # [T, D]
    MK = batch_dec["mask"][node_idx]              # [T, D] (1 where observed)

    mu0  = z0_mu[0, node_idx]                     # [Z]
    z0s  = z0_samples[sample_id, node_idx]        # [Z]
    zt   = z_traj[sample_id, node_idx]            # [T, Z]

    # pick feature & latent dim
    mu_feat = _np(MU[:, feat_idx])                # [T]
    y_feat  = _np(Y[:,  feat_idx])                # [T]
    mk_feat = _np((MK[:, feat_idx] > 0.5))        # [T] bool
    z_lat   = _np(zt[:, latent_dim_to_plot])      # [T]
    mu0_dim = float(mu0[latent_dim_to_plot].item())
    z0_dim  = float(z0s[latent_dim_to_plot].item())

    # ---------- plot ----------
    os.makedirs(outdir, exist_ok=True)
    title_bits = [tag]
    if epoch is not None: title_bits.append(f"ep{epoch}")
    if step  is not None: title_bits.append(f"it{step}")
    title_bits.append(f"node{node_idx}/feat{feat_idx}/z{latent_dim_to_plot}")
    ttl = " • ".join(title_bits)

    plt.figure(figsize=(9, 7.5), dpi=160)

    # (1) decoded vs truth
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t_real, mu_feat, "--", lw=2.0, label="pred mean")
    if mk_feat.any():
        ax1.scatter(t_real[mk_feat], y_feat[mk_feat], s=20, label="truth (obs)")
        ax1.scatter(t_real[mk_feat], mu_feat[mk_feat], s=28, marker="x", label="pred @ obs-time")
    ax1.set_ylabel(f"decoded f{feat_idx}")
    ax1.grid(alpha=0.25, lw=0.6); ax1.legend(frameon=False, fontsize=9)

    # (2) latent trajectory z(t) (one dim)
    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.plot(t_real, z_lat, lw=1.8)
    ax2.axhline(mu0_dim, ls="--", lw=1.2, alpha=0.6, label="z0 μ (same dim)")
    ax2.set_ylabel(f"z[{latent_dim_to_plot}]")
    ax2.grid(alpha=0.25, lw=0.6); ax2.legend(frameon=False, fontsize=9)

    # (3) z0 per-dim (bar) + sample marker (optional)
    ax3 = plt.subplot(3,1,3)
    mu0_np = _np(mu0)
    ax3.bar(np.arange(mu0_np.size), mu0_np, alpha=0.55)
    ax3.scatter([latent_dim_to_plot], [z0_dim], s=35, c="k", label="sampled z0 (this dim)")
    ax3.set_xlabel("latent dim"); ax3.set_ylabel("z0 μ")
    ax3.grid(axis="y", alpha=0.25, lw=0.6); ax3.legend(frameon=False, fontsize=9)

    plt.suptitle(ttl); plt.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = os.path.join(outdir, f"{tag}_node{node_idx}_feat{feat_idx}_z{latent_dim_to_plot}.png")
    plt.savefig(outpath, bbox_inches="tight"); plt.close()
    return outpath

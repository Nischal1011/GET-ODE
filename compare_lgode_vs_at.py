#!/usr/bin/env python3
import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


RE_TEST_MSE = re.compile(r"\[Test.*?\]\s*\|\s*Loss.*?\|\s*MSE\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)
RE_TEST_MSE_ALT = re.compile(r"\[Test.*?\]\s*\|\s.*?MSE\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)
RE_EPOCH = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)


def run_one_model(
    script_path: Path,
    args_list: list[str],
    workdir: Path,
    env: dict | None = None,
) -> tuple[list[int], list[float], str]:
    """
    Runs a training script, captures stdout, and parses test MSE per epoch.
    Returns: (epochs, mses, raw_stdout)
    """
    cmd = [sys.executable, str(script_path)] + args_list
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("CWD:", str(workdir))
    print("=" * 100)

    p = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    raw_lines = []
    epochs = []
    mses = []

    current_epoch = None
    for line in p.stdout:
        raw_lines.append(line)
        sys.stdout.write(line)  # stream to console
        sys.stdout.flush()

        # Track epoch
        m_ep = RE_EPOCH.search(line)
        if m_ep:
            try:
                current_epoch = int(m_ep.group(1))
            except Exception:
                pass

        # Parse test mse from test line
        m = RE_TEST_MSE.search(line) or RE_TEST_MSE_ALT.search(line)
        if m:
            try:
                mse_val = float(m.group(1))
                # use current_epoch if available
                if current_epoch is None:
                    # fallback: append sequentially
                    epochs.append(len(epochs) + 1)
                else:
                    epochs.append(current_epoch)
                mses.append(mse_val)
            except Exception:
                pass

    rc = p.wait()
    raw_stdout = "".join(raw_lines)

    if rc != 0:
        print("\n[ERROR] Training script exited with code:", rc)
        # still return what we parsed (useful for debugging)
        return epochs, mses, raw_stdout

    if len(mses) == 0:
        print("\n[WARN] No test MSE parsed from output. Make sure your script prints a line like:")
        print("  Epoch 0001 [Test ...] | ... | MSE <value> | ...")
    return epochs, mses, raw_stdout


def plot_mse_vs_epoch(outpath: Path, curves: dict[str, tuple[list[int], list[float]]], title: str):
    plt.figure()
    for name, (ep, mse) in curves.items():
        plt.plot(ep, mse, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"[plot] wrote: {outpath}")


def plot_mse_vs_percent(outpath: Path, curves: dict[str, tuple[list[float], list[float]]], title: str):
    plt.figure()
    for name, (pct, mse) in curves.items():
        plt.plot(pct, mse, marker="o", label=name)
    plt.xlabel("Sample percent (raw=test)")
    plt.ylabel("Best test MSE (over epochs)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"[plot] wrote: {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lg_root", type=str, default="LG-ODE", help="Folder containing vanilla LG-ODE code")
    ap.add_argument("--at_root", type=str, default="AT_LG-ODE", help="Folder containing AT LG-ODE code")

    ap.add_argument("--lg_script", type=str, default="run_models.py", help="Vanilla training script filename")
    ap.add_argument("--at_script", type=str, default="run_models.py", help="AT training script filename")

    ap.add_argument("--datasets", type=str, default="spring,charged,motion", help="Comma list: spring,charged,motion")
    # ap.add_argument("--datasets", type=str, default="spring", help="Comma list: spring,charged,motion")

    ap.add_argument("--mode", type=str, default="interp", choices=["interp", "extrap"])
    ap.add_argument("--niters", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1991)
    ap.add_argument("--solver", type=str, default="rk4")
    ap.add_argument("--odenet", type=str, default="NRI")
    ap.add_argument("--z0_encoder", type=str, default="GTrans")
    ap.add_argument("--rec_attention", type=str, default="attention")

    # simple compare at one percent
    ap.add_argument("--sample_percent", type=float, default=0.6)

    # sweep mode
    ap.add_argument("--sweep_percent", action="store_true", help="If set, sweep sample percents and plot MSE vs percent")
    ap.add_argument("--percent_list", type=str, default="0.2,0.4,0.6,0.8",
                    help="Comma list percents for sweep, e.g. 0.1,0.2,0.4,0.6,0.8")

    # output
    ap.add_argument("--outdir", type=str, default="compare_plots")

    # AT-only knobs (will be passed to AT script; vanilla ignores unknown args if you keep them separate)
    ap.add_argument("--K_lag", type=int, default=16)
    ap.add_argument("--attn_v", type=float, default=1.0)
    ap.add_argument("--gumbel_tau", type=float, default=1.0)
    ap.add_argument("--gumbel_hard", type=str, default="False")
    ap.add_argument("--edge_prior_mode", type=str, default="graph")
    ap.add_argument("--edge_prior_p", type=float, default=0.9)
    ap.add_argument("--kl_edge_coef", type=float, default=1.0)

    args = ap.parse_args()

    lg_root = Path(args.lg_root).resolve()
    at_root = Path(args.at_root).resolve()
    lg_script = (lg_root / args.lg_script).resolve()
    at_script = (at_root / args.at_script).resolve()

    if not lg_script.exists():
        raise FileNotFoundError(f"Vanilla script not found: {lg_script}")
    if not at_script.exists():
        raise FileNotFoundError(f"AT script not found: {at_script}")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    extrap_flag = "True" if args.mode == "extrap" else "False"

    # Base args common to both scripts
    base_common = [
        "--niters", str(args.niters),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--random-seed", str(args.seed),
        "--solver", str(args.solver),
        "--odenet", str(args.odenet),
        "--z0-encoder", str(args.z0_encoder),
        "--rec_attention", str(args.rec_attention),
        "--extrap", extrap_flag,
    ]

    # AT args
    at_only = [
        "--K_lag", str(args.K_lag),
        "--attn_v", str(args.attn_v),
        "--gumbel_tau", str(args.gumbel_tau),
        "--gumbel_hard", str(args.gumbel_hard),
        "--edge_prior_mode", str(args.edge_prior_mode),
        "--edge_prior_p", str(args.edge_prior_p),
        "--kl_edge_coef", str(args.kl_edge_coef),
    ]

    if not args.sweep_percent:
        # ========= Single run: plot MSE vs epoch =========
        for data_name in datasets:
            sp = args.sample_percent

            lg_args = base_common + [
                "--raw", data_name,
                "--sample-percent-raw", str(sp),
                "--sample-percent-test", str(sp),
                "--alias", f"LG_{data_name}_p{sp}",
            ]
            at_args = base_common + at_only + [
                "--raw", data_name,
                "--sample-percent-raw", str(sp),
                "--sample-percent-test", str(sp),
                "--alias", f"AT_{data_name}_p{sp}",
            ]

            ep_lg, mse_lg, _ = run_one_model(lg_script, lg_args, workdir=lg_root)
            ep_at, mse_at, _ = run_one_model(at_script, at_args, workdir=at_root)

            title = f"{data_name.upper()} • mode={args.mode} • sample%={sp}"
            plot_path = outdir / f"mse_vs_epoch_{data_name}_mode_{args.mode}_p{sp}.png"
            plot_mse_vs_epoch(
                plot_path,
                curves={
                    "LG-ODE": (ep_lg, mse_lg),
                    "AT-LG-ODE": (ep_at, mse_at),
                },
                title=title,
            )

    else:
        # ========= Sweep: plot best MSE vs sample percent =========
        percents = [float(x.strip()) for x in args.percent_list.split(",") if x.strip()]
        for data_name in datasets:
            lg_best = []
            at_best = []

            for sp in percents:
                lg_args = base_common + [
                    "--raw", data_name,
                    "--sample-percent-raw", str(sp),
                    "--sample-percent-test", str(sp),
                    "--alias", f"LG_{data_name}_p{sp}",
                ]
                at_args = base_common + at_only + [
                    "--raw", data_name,
                    "--sample-percent-raw", str(sp),
                    "--sample-percent-test", str(sp),
                    "--alias", f"AT_{data_name}_p{sp}",
                ]

                ep_lg, mse_lg, _ = run_one_model(lg_script, lg_args, workdir=lg_root)
                ep_at, mse_at, _ = run_one_model(at_script, at_args, workdir=at_root)

                lg_best.append(min(mse_lg) if len(mse_lg) else float("nan"))
                at_best.append(min(mse_at) if len(mse_at) else float("nan"))

            title = f"{data_name.upper()} • mode={args.mode} • best test MSE vs sample%"
            plot_path = outdir / f"mse_vs_percent_{data_name}_mode_{args.mode}.png"
            plot_mse_vs_percent(
                plot_path,
                curves={
                    "LG-ODE": (percents, lg_best),
                    "AT-LG-ODE": (percents, at_best),
                },
                title=title,
            )


if __name__ == "__main__":
    main()


    # run these script:



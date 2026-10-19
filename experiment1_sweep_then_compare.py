#!/usr/bin/env python3
import os
import re
import sys
import json
import time
import argparse
import itertools
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


# ----------------------------
# Regex for parsing logs
# ----------------------------
RE_EPOCH = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)

# Matches: "... | MSE 0.123" or "... MSE 0.123"
RE_MSE = re.compile(r"\bMSE\b[^0-9\-]*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Matches: "Epoch 003 | Test MSE 0.123"
RE_TEST_MSE_SIMPLE = re.compile(r"Epoch\s+(\d+)\s*\|\s*Test\s+MSE\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass
class RunResult:
    epochs: List[int]
    test_mse: List[float]
    best_mse: float
    best_epoch: int
    raw_log_path: Path


def run_and_parse(
    workdir: Path,
    script: Path,
    args_list: List[str],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
) -> RunResult:
    """
    Runs a training script as subprocess, saves stdout+stderr to log_path,
    parses test MSE per epoch.
    """
    cmd = [sys.executable, str(script)] + args_list
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.write("CWD: " + str(workdir) + "\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        epochs: List[int] = []
        mses: List[float] = []
        current_epoch: Optional[int] = None

        for line in p.stdout:
            f.write(line)
            f.flush()

            # Epoch tracking
            m_ep = RE_EPOCH.search(line)
            if m_ep:
                try:
                    current_epoch = int(m_ep.group(1))
                except Exception:
                    pass

            # Prefer explicit "Epoch | Test MSE X" format if present
            m_simple = RE_TEST_MSE_SIMPLE.search(line)
            if m_simple:
                ep = int(m_simple.group(1))
                mse = float(m_simple.group(2))
                epochs.append(ep)
                mses.append(mse)
                continue

            # Otherwise parse generic MSE (assumes it's test line if printed in test block)
            m_mse = RE_MSE.search(line)
            if m_mse and ("Test" in line or "[Test" in line or "test" in line):
                try:
                    mse = float(m_mse.group(1))
                    ep = current_epoch if current_epoch is not None else (len(epochs) + 1)
                    epochs.append(ep)
                    mses.append(mse)
                except Exception:
                    pass

        rc = p.wait()

    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}). See log: {log_path}")

    if not mses:
        raise RuntimeError(f"No test MSE parsed. Check log format: {log_path}")

    best_mse = min(mses)
    best_idx = mses.index(best_mse)
    best_epoch = epochs[best_idx]

    return RunResult(epochs=epochs, test_mse=mses, best_mse=best_mse, best_epoch=best_epoch, raw_log_path=log_path)


def save_plot_mse_vs_epoch(
    out_pdf: Path,
    curves: Dict[str, Tuple[List[int], List[float]]],
    title: str,
    hide_axis_labels: bool = True,
):
    plt.figure()
    for name, (ep, mse) in curves.items():
        plt.plot(ep, mse, marker="o", label=name)

    if not hide_axis_labels:
        plt.xlabel("Epoch")
        plt.ylabel("Test MSE")

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[plot] wrote {out_pdf}")


def write_table_csv(out_csv: Path, rows: List[Dict[str, object]]):
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[table] wrote {out_csv}")


def default_repo_layout(root: Path) -> Dict[str, Dict[str, str]]:
    """
    Edit these paths ONCE if your scripts are named differently.
    """
    return {
        "LG-ODE": {
            "workdir": str(root / "LG-ODE"),
            "script": "run_models.py",
        },
        "AT-ODE": {
            "workdir": str(root / "AT_LG-ODE"),
            "script": "run_models.py",
        },
        # If your CG-ODE folder has a different script name, change it here.
        "CG-ODE": {
            "workdir": str(root / "CG-ODE-main"),
            "script": "run_models.py",
        },
        # CSG-ODE is your own implementation folder. Adjust script as needed.
        "CSG-ODE": {
            "workdir": str(root / "CSG-ODE"),
            "script": "run_csgode_springs.py",  # change if different
        },
    }


def make_common_args(dataset: str, mode: str, obs_ratio: float, niters: int, batch_size: int, seed: int) -> List[str]:
    # For LG-ODE style scripts
    extrap_flag = "True" if mode == "extrap" else "False"
    return [
        "--raw", dataset,
        "--extrap", extrap_flag,
        "--niters", str(niters),
        "--batch-size", str(batch_size),
        "--random-seed", str(seed),
        "--sample-percent-raw", str(obs_ratio),
        "--sample-percent-test", str(obs_ratio),
    ]


def make_at_args(at_cfg: Dict[str, object]) -> List[str]:
    # AT-only args; your AT script supports these.
    args = []
    for k, v in at_cfg.items():
        args += [f"--{k}", str(v)]
    return args


def make_csg_args(dataset: str, mode: str, obs_ratio: float, niters: int, batch_size: int, seed: int) -> List[str]:
    # For the minimal CSG script I gave you earlier
    # (If your CSG runner uses different flags, edit here.)
    return [
        "--mode", mode,
        "--obs_ratio", str(obs_ratio),
        "--epochs", str(niters),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
    ]


def sweep_space_default() -> List[Dict[str, object]]:
    """
    A small, efficient sweep. Expand later.
    """
    grid = []
    for K_lag, attn_v, gumbel_tau, edge_prior_p, kl_edge_coef in itertools.product(
        [8, 16],
        [0.5, 1.0],
        [0.8, 1.0],
        [0.7, 0.9],
        [0.1, 1.0],
    ):
        grid.append({
            "K_lag": K_lag,
            "attn_v": attn_v,
            "gumbel_tau": gumbel_tau,
            "gumbel_hard": "False",
            "edge_prior_mode": "graph",
            "edge_prior_p": edge_prior_p,
            "kl_edge_coef": kl_edge_coef,
        })
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Folder containing AT_LG-ODE, LG-ODE, CG-ODE-main, CSG-ODE")
    ap.add_argument("--outdir", type=str, default="exp1_results")

    ap.add_argument("--datasets", type=str, default="spring,charged", help="Comma list. Add kuramoto,gene if you have scripts.")
    ap.add_argument("--niters", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1991)

    ap.add_argument("--obs_ratio", type=float, default=0.6)
    ap.add_argument("--do_sweep", action="store_true", help="If set, hyperparam-sweep AT per dataset.")
    ap.add_argument("--max_sweep", type=int, default=12, help="Limit number of sweep trials for speed.")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    repos = default_repo_layout(root)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # record best AT configs per dataset+mode
    best_cfgs: Dict[str, Dict[str, Dict[str, object]]] = {}

    sweep_grid = sweep_space_default()

    for dataset in datasets:
        best_cfgs[dataset] = {}
        for mode in ["interp", "extrap"]:
            # --------------------------------------
            # 1) Hyperparam sweep for AT-ODE (optional)
            # --------------------------------------
            if args.do_sweep:
                print(f"\n=== SWEEP AT-ODE | dataset={dataset} mode={mode} obs={args.obs_ratio} ===")
                best = None
                best_cfg = None

                trials = sweep_grid[: max(1, min(args.max_sweep, len(sweep_grid)))]

                for ti, cfg in enumerate(trials):
                    log_path = outdir / "logs" / dataset / mode / f"AT_sweep_{ti:03d}.log"
                    at_workdir = Path(repos["AT-ODE"]["workdir"])
                    at_script = at_workdir / repos["AT-ODE"]["script"]

                    at_args = make_common_args(dataset, mode, args.obs_ratio, args.niters, args.batch_size, args.seed)
                    at_args += make_at_args(cfg)
                    # unique alias avoids overwriting logs/checkpoints if your script uses it
                    at_args += ["--alias", f"AT_SWEEP_{dataset}_{mode}_{ti:03d}"]

                    rr = run_and_parse(at_workdir, at_script, at_args, log_path)
                    if best is None or rr.best_mse < best:
                        best = rr.best_mse
                        best_cfg = cfg
                        print(f"[best so far] mse={best:.6f} cfg={best_cfg}")

                assert best_cfg is not None
                best_cfgs[dataset][mode] = best_cfg
            else:
                # if not sweeping, use a single default config
                best_cfgs[dataset][mode] = {
                    "K_lag": 16,
                    "attn_v": 1.0,
                    "gumbel_tau": 1.0,
                    "gumbel_hard": "False",
                    "edge_prior_mode": "graph",
                    "edge_prior_p": 0.9,
                    "kl_edge_coef": 1.0,
                }

            # Save best cfg
            (outdir / "best_cfgs").mkdir(parents=True, exist_ok=True)
            with open(outdir / "best_cfgs" / f"best_at_{dataset}_{mode}.json", "w") as f:
                json.dump(best_cfgs[dataset][mode], f, indent=2)

            # --------------------------------------
            # 2) Compare models for this dataset/mode
            # --------------------------------------
            print(f"\n=== COMPARE | dataset={dataset} mode={mode} obs={args.obs_ratio} ===")
            rows = []
            curves = {}

            # LG-ODE
            lg_workdir = Path(repos["LG-ODE"]["workdir"])
            lg_script = lg_workdir / repos["LG-ODE"]["script"]
            lg_args = make_common_args(dataset, mode, args.obs_ratio, args.niters, args.batch_size, args.seed) + \
                      ["--alias", f"LG_{dataset}_{mode}"]
            lg_log = outdir / "logs" / dataset / mode / "LG_ODE.log"
            lg_rr = run_and_parse(lg_workdir, lg_script, lg_args, lg_log)
            rows.append({"dataset": dataset, "mode": mode, "model": "LG-ODE", "best_mse": lg_rr.best_mse, "best_epoch": lg_rr.best_epoch})
            curves["LG-ODE"] = (lg_rr.epochs, lg_rr.test_mse)

            # AT-ODE
            at_workdir = Path(repos["AT-ODE"]["workdir"])
            at_script = at_workdir / repos["AT-ODE"]["script"]
            at_cfg = best_cfgs[dataset][mode]
            at_args = make_common_args(dataset, mode, args.obs_ratio, args.niters, args.batch_size, args.seed)
            at_args += make_at_args(at_cfg)
            at_args += ["--alias", f"AT_{dataset}_{mode}"]
            at_log = outdir / "logs" / dataset / mode / "AT_ODE.log"
            at_rr = run_and_parse(at_workdir, at_script, at_args, at_log)
            rows.append({"dataset": dataset, "mode": mode, "model": "AT-ODE", "best_mse": at_rr.best_mse, "best_epoch": at_rr.best_epoch, **{f"at_{k}": v for k, v in at_cfg.items()}})
            curves["AT-ODE"] = (at_rr.epochs, at_rr.test_mse)

            # CG-ODE (if script exists)
            try:
                cg_workdir = Path(repos["CG-ODE"]["workdir"])
                cg_script = cg_workdir / repos["CG-ODE"]["script"]
                cg_args = make_common_args(dataset, mode, args.obs_ratio, args.niters, args.batch_size, args.seed) + \
                          ["--alias", f"CG_{dataset}_{mode}"]
                cg_log = outdir / "logs" / dataset / mode / "CG_ODE.log"
                cg_rr = run_and_parse(cg_workdir, cg_script, cg_args, cg_log)
                rows.append({"dataset": dataset, "mode": mode, "model": "CG-ODE", "best_mse": cg_rr.best_mse, "best_epoch": cg_rr.best_epoch})
                curves["CG-ODE"] = (cg_rr.epochs, cg_rr.test_mse)
            except Exception as e:
                rows.append({"dataset": dataset, "mode": mode, "model": "CG-ODE", "best_mse": None, "note": f"skip: {repr(e)}"})

            # CSG-ODE (only guaranteed for springs unless you implemented charged etc.)
            try:
                csg_workdir = Path(repos["CSG-ODE"]["workdir"])
                csg_script = csg_workdir / repos["CSG-ODE"]["script"]
                csg_args = make_csg_args(dataset, mode, args.obs_ratio, args.niters, args.batch_size, args.seed)
                csg_log = outdir / "logs" / dataset / mode / "CSG_ODE.log"
                csg_rr = run_and_parse(csg_workdir, csg_script, csg_args, csg_log)
                rows.append({"dataset": dataset, "mode": mode, "model": "CSG-ODE", "best_mse": csg_rr.best_mse, "best_epoch": csg_rr.best_epoch})
                curves["CSG-ODE"] = (csg_rr.epochs, csg_rr.test_mse)
            except Exception as e:
                rows.append({"dataset": dataset, "mode": mode, "model": "CSG-ODE", "best_mse": None, "note": f"skip: {repr(e)}"})

            # Save table
            out_csv = outdir / "tables" / f"{dataset}_{mode}_obs{args.obs_ratio}.csv"
            write_table_csv(out_csv, rows)
            out_json = outdir / "tables" / f"{dataset}_{mode}_obs{args.obs_ratio}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(rows, f, indent=2)

            # Plot MSE vs epoch
            title = f"{dataset} | {mode} | obs={args.obs_ratio}"
            out_pdf = outdir / "plots" / f"mse_vs_epoch_{dataset}_{mode}_obs{args.obs_ratio}.pdf"
            save_plot_mse_vs_epoch(out_pdf, curves, title, hide_axis_labels=True)

    print("\nDone. Results in:", outdir)


if __name__ == "__main__":
    main()

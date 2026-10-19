#!/usr/bin/env python3
import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

RE_MSE = re.compile(r"\bMSE\b[^0-9\-]*([0-9]*\.?[0-9]+)", re.IGNORECASE)
RE_TEST_MSE_SIMPLE = re.compile(r"Epoch\s+(\d+)\s*\|\s*Test\s+MSE\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)


def run_best_mse(workdir: Path, script: Path, args_list: List[str], log_path: Path) -> float:
    cmd = [sys.executable, str(script)] + args_list
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best = None
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
        )
        for line in p.stdout:
            f.write(line)
            f.flush()

            m_simple = RE_TEST_MSE_SIMPLE.search(line)
            if m_simple:
                mse = float(m_simple.group(2))
                best = mse if best is None else min(best, mse)
                continue

            if ("Test" in line or "[Test" in line or "test" in line) and ("MSE" in line):
                m = RE_MSE.search(line)
                if m:
                    mse = float(m.group(1))
                    best = mse if best is None else min(best, mse)

        rc = p.wait()

    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}) log={log_path}")
    if best is None:
        raise RuntimeError(f"No test MSE parsed. log={log_path}")
    return best


def default_repo_layout(root: Path) -> Dict[str, Dict[str, str]]:
    return {
        "LG-ODE": {"workdir": str(root / "LG-ODE"), "script": "run_models.py"},
        "AT-ODE": {"workdir": str(root / "AT_LG-ODE"), "script": "run_models.py"},
        "CG-ODE": {"workdir": str(root / "CG-ODE-main"), "script": "run_models.py"},
        "CSG-ODE": {"workdir": str(root / "CSG-ODE"), "script": "run_csgode_springs.py"},
    }


def make_common_args(dataset: str, mode: str, obs_ratio: float, niters: int, batch_size: int, seed: int) -> List[str]:
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
    out = []
    for k, v in at_cfg.items():
        out += [f"--{k}", str(v)]
    return out


def make_csg_args(dataset: str, mode: str, obs_ratio: float, niters: int, batch_size: int, seed: int) -> List[str]:
    return [
        "--mode", mode,
        "--obs_ratio", str(obs_ratio),
        "--epochs", str(niters),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
    ]


def plot_mse_vs_obs(out_pdf: Path, obs: List[float], curves: Dict[str, List[float]], title: str):
    plt.figure()
    for name, ys in curves.items():
        plt.plot(obs, ys, marker="o", label=name)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    # no axis labels as requested
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"[plot] wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--outdir", type=str, default="exp2_results")
    ap.add_argument("--dataset", type=str, default="spring")
    ap.add_argument("--obs_list", type=str, default="0.2,0.4,0.8,1.0")
    ap.add_argument("--niters", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1991)

    ap.add_argument("--best_cfg_dir", type=str, default="exp1_results/best_cfgs",
                    help="Folder containing best_at_<dataset>_<mode>.json from experiment 1")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    repos = default_repo_layout(root)

    obs = [float(x.strip()) for x in args.obs_list.split(",") if x.strip()]

    for mode in ["interp", "extrap"]:
        # load best cfg for this dataset/mode
        cfg_path = Path(args.best_cfg_dir) / f"best_at_{args.dataset}_{mode}.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing best cfg file: {cfg_path}")
        at_cfg = json.loads(cfg_path.read_text())

        curves: Dict[str, List[float]] = {"LG-ODE": [], "AT-ODE": [], "CG-ODE": [], "CSG-ODE": []}

        for sp in obs:
            print(f"\n=== {args.dataset} | {mode} | obs={sp} ===")

            # LG-ODE
            lg_wd = Path(repos["LG-ODE"]["workdir"])
            lg_sc = lg_wd / repos["LG-ODE"]["script"]
            lg_args = make_common_args(args.dataset, mode, sp, args.niters, args.batch_size, args.seed) + ["--alias", f"LG_{sp}"]
            lg_log = outdir / "logs" / mode / f"LG_obs{sp}.log"
            curves["LG-ODE"].append(run_best_mse(lg_wd, lg_sc, lg_args, lg_log))

            # AT-ODE
            at_wd = Path(repos["AT-ODE"]["workdir"])
            at_sc = at_wd / repos["AT-ODE"]["script"]
            at_args = make_common_args(args.dataset, mode, sp, args.niters, args.batch_size, args.seed)
            at_args += make_at_args(at_cfg)
            at_args += ["--alias", f"AT_{sp}"]
            at_log = outdir / "logs" / mode / f"AT_obs{sp}.log"
            curves["AT-ODE"].append(run_best_mse(at_wd, at_sc, at_args, at_log))

            # CG-ODE
            try:
                cg_wd = Path(repos["CG-ODE"]["workdir"])
                cg_sc = cg_wd / repos["CG-ODE"]["script"]
                cg_args = make_common_args(args.dataset, mode, sp, args.niters, args.batch_size, args.seed) + ["--alias", f"CG_{sp}"]
                cg_log = outdir / "logs" / mode / f"CG_obs{sp}.log"
                curves["CG-ODE"].append(run_best_mse(cg_wd, cg_sc, cg_args, cg_log))
            except Exception:
                curves["CG-ODE"].append(float("nan"))

            # CSG-ODE (may only be valid for springs unless extended)
            try:
                csg_wd = Path(repos["CSG-ODE"]["workdir"])
                csg_sc = csg_wd / repos["CSG-ODE"]["script"]
                csg_args = make_csg_args(args.dataset, mode, sp, args.niters, args.batch_size, args.seed)
                csg_log = outdir / "logs" / mode / f"CSG_obs{sp}.log"
                curves["CSG-ODE"].append(run_best_mse(csg_wd, csg_sc, csg_args, csg_log))
            except Exception:
                curves["CSG-ODE"].append(float("nan"))

        # Plot
        out_pdf = outdir / f"mse_vs_obs_{args.dataset}_{mode}.pdf"
        title = f"{args.dataset} | {mode}"
        plot_mse_vs_obs(out_pdf, obs, curves, title)

        # Save numbers
        out_json = outdir / f"numbers_{args.dataset}_{mode}.json"
        with open(out_json, "w") as f:
            json.dump({"obs": obs, "curves": curves, "at_cfg": at_cfg}, f, indent=2)
        print(f"[raw] wrote {out_json}")


if __name__ == "__main__":
    main()

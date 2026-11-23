#!/usr/bin/env python3
"""Generate, execute, and aggregate the CSG-ODE reproduction matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lib.baselines.csg_ode.config import canonical_dataset_name
from lib.baselines.csg_ode.data_adapter import audit_pems_feature_semantics
from lib.baselines.csg_ode.diagnostics import environment_record


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent

PAPER_TABLE_SCALED_MSE = {
    "springs": {
        "interp": {0.4: 0.1550, 0.6: 0.1440, 0.8: 0.1386},
        "extrap": {0.4: 1.3495, 0.6: 1.2969, 0.8: 1.2691},
    },
    "charged": {
        "interp": {0.4: 0.7947, 0.6: 0.7169, 0.8: 0.7099},
        "extrap": {0.4: 5.5086, 0.6: 4.7690, 0.8: 4.4966},
    },
    "motion_walk": {
        "interp": {0.4: 0.0439, 0.6: 0.0406, 0.8: 0.0400},
        "extrap": {0.4: 0.1791, 0.6: 0.1539, 0.8: 0.1593},
    },
    "pems08": {
        "interp": {0.4: 0.2526, 0.6: 0.2827, 0.8: 0.3360},
        "extrap": {0.4: 1.6607, 0.6: 1.7436, 0.8: 1.4937},
    },
}

VARIANT_FLAGS = {
    "full": None,
    "aq": "--ablation-aq",
    "no-ei": "--ablation-no-ei",
    "no-g": "--ablation-no-g",
    "no-ni": "--ablation-no-ni",
}

AGGREGATE_FIELDS = [
    "dataset",
    "task",
    "obs_ratio",
    "variant",
    "completed_seeds",
    "expected_seeds",
    "complete_seed_set",
    "raw_mse_mean",
    "raw_mse_std",
    "table_scaled_mse_mean",
    "table_scaled_mse_std",
    "best_seed_mse",
    "best_seed",
    "paper_table_scaled_mse",
    "paper_raw_mse",
    "relative_error_to_paper",
    "paper_tolerance",
    "within_paper_tolerance",
    "mean_training_seconds",
]


@dataclass(frozen=True)
class SweepRun:
    dataset: str
    task: str
    obs_ratio: float
    seed: int
    variant: str
    run_name: str
    command: list[str]
    result_path: str


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="springs,charged,motion_walk,pems08")
    parser.add_argument("--tasks", default="interp,extrap")
    parser.add_argument("--obs-ratios", default="0.4,0.6,0.8")
    parser.add_argument("--seeds", default="1991,42,7")
    parser.add_argument("--variants", default="full")
    parser.add_argument("--normalization-mode", default="paper_splitwise_maxabs")
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "experiments")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--wandb-mode", choices=["disabled", "offline", "online"], default="disabled"
    )
    parser.add_argument("--wandb-project", default="at-ode")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-dataset-mismatch", action="store_true")
    return parser.parse_args()


def build_runs(args: argparse.Namespace) -> list[SweepRun]:
    datasets = [canonical_dataset_name(value) for value in comma_values(args.datasets)]
    tasks = comma_values(args.tasks)
    ratios = [float(value) for value in comma_values(args.obs_ratios)]
    seeds = [int(value) for value in comma_values(args.seeds)]
    variants = comma_values(args.variants)
    unknown_variants = sorted(set(variants) - set(VARIANT_FLAGS))
    if unknown_variants:
        raise ValueError(f"Unknown variants: {', '.join(unknown_variants)}")
    unknown_tasks = sorted(set(tasks) - {"interp", "extrap"})
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {', '.join(unknown_tasks)}")
    unknown_ratios = sorted(set(ratios) - {0.4, 0.6, 0.8})
    if unknown_ratios:
        raise ValueError(f"Unsupported observation ratios: {unknown_ratios}")

    runs: list[SweepRun] = []
    for dataset in datasets:
        config_path = SCRIPT_DIR / "configs" / "csg_ode" / f"{dataset}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        for task in tasks:
            for ratio in ratios:
                for seed in seeds:
                    for variant in variants:
                        run_name = (
                            f"csgode_{dataset}_{task}_{ratio:g}_seed{seed}_{variant}_"
                            f"{args.normalization_mode}"
                        )
                        command = [
                            sys.executable,
                            str(SCRIPT_DIR / "run_csg_ode.py"),
                            "--model",
                            "csg_ode",
                            "--dataset",
                            dataset,
                            "--task",
                            task,
                            "--obs-ratio",
                            str(ratio),
                            "--seed",
                            str(seed),
                            "--config",
                            str(config_path),
                            "--normalization-mode",
                            args.normalization_mode,
                            "--data-root",
                            str(args.data_root),
                            "--output-root",
                            str(args.output_root),
                            "--run-name",
                            run_name,
                            "--device",
                            args.device,
                            "--wandb-mode",
                            args.wandb_mode,
                            "--wandb-project",
                            args.wandb_project,
                        ]
                        flag = VARIANT_FLAGS[variant]
                        if flag:
                            command.append(flag)
                        if args.allow_dataset_mismatch:
                            command.append("--allow-dataset-mismatch")
                        command.extend(shlex.split(args.extra_args))
                        runs.append(
                            SweepRun(
                                dataset,
                                task,
                                ratio,
                                seed,
                                variant,
                                run_name,
                                command,
                                str(args.output_root / run_name / "result.json"),
                            )
                        )
    return runs


def aggregate(runs: list[SweepRun], output_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = {}
    for run in runs:
        result_path = Path(run.result_path)
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        grouped.setdefault((run.dataset, run.task, run.obs_ratio, run.variant), []).append(result)

    rows: list[dict[str, Any]] = []
    expected_seeds = len({run.seed for run in runs})
    for (dataset, task, ratio, variant), results in sorted(grouped.items()):
        values = [float(result["test"]["mse_lgode"]) for result in results]
        scaled = [value * 100.0 for value in values]
        paper = PAPER_TABLE_SCALED_MSE[dataset][task][ratio] if variant == "full" else None
        mean_scaled = statistics.mean(scaled)
        best_result = min(results, key=lambda result: float(result["test"]["mse_lgode"]))
        complete_seed_set = len(values) == expected_seeds
        relative_error = abs(mean_scaled - paper) / paper if paper is not None else None
        tolerance = 0.15 if dataset == "pems08" else 0.10
        row = {
            "dataset": dataset,
            "task": task,
            "obs_ratio": ratio,
            "variant": variant,
            "completed_seeds": len(values),
            "expected_seeds": expected_seeds,
            "complete_seed_set": complete_seed_set,
            "raw_mse_mean": statistics.mean(values),
            "raw_mse_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "table_scaled_mse_mean": mean_scaled,
            "table_scaled_mse_std": statistics.stdev(scaled) if len(scaled) > 1 else 0.0,
            "best_seed_mse": min(values),
            "best_seed": int(best_result["config"]["seed"]),
            "paper_table_scaled_mse": paper,
            "paper_raw_mse": paper * 1e-2 if paper is not None else None,
            "relative_error_to_paper": relative_error,
            "paper_tolerance": tolerance if paper is not None else None,
            "within_paper_tolerance": (
                complete_seed_set and relative_error <= tolerance
                if relative_error is not None
                else None
            ),
            "mean_training_seconds": statistics.mean(
                float(result["training_seconds"]) for result in results
            ),
        }
        rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "aggregate.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_root / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return rows


def main() -> int:
    args = parse_args()
    runs = build_runs(args)
    pems_requested = any(run.dataset == "pems08" for run in runs)
    pems_preflight = audit_pems_feature_semantics(args.data_root) if pems_requested else None
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "environment": environment_record(REPOSITORY_ROOT),
        "execute": args.execute,
        "dataset_preflight": {"pems08": pems_preflight} if pems_preflight else {},
        "runs": [asdict(run) for run in runs],
    }
    (args.output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    if pems_preflight is not None and not pems_preflight["paper_compatible"]:
        print(f"PEMS08 preflight: {pems_preflight['reason']}", file=sys.stderr)
        if args.execute and not args.collect_only and not args.allow_dataset_mismatch:
            print(
                "No jobs were launched. Correct the PEMS08 source or explicitly pass "
                "--allow-dataset-mismatch for excluded plumbing runs.",
                file=sys.stderr,
            )
            return 2

    if not args.collect_only:
        for index, run in enumerate(runs, start=1):
            printable = " ".join(shlex.quote(value) for value in run.command)
            print(f"[{index}/{len(runs)}] {printable}")
            if args.execute:
                completed = subprocess.run(run.command, cwd=SCRIPT_DIR, check=False)
                if completed.returncode and not args.continue_on_error:
                    return completed.returncode

    rows = aggregate(runs, args.output_root)
    completed = sum(row["completed_seeds"] for row in rows)
    print(f"Prepared {len(runs)} runs; collected {completed} completed seed results.")
    print(f"Manifest: {args.output_root / 'sweep_manifest.json'}")
    print(f"Aggregate: {args.output_root / 'aggregate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train and evaluate the isolated CSG-ODE baseline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import torch

from lib.baselines.csg_ode.checkpoint import load_checkpoint, save_checkpoint
from lib.baselines.csg_ode.communicability import CommunicabilityCache
from lib.baselines.csg_ode.config import load_config, paper_config
from lib.baselines.csg_ode.data_adapter import build_datasets, make_loaders
from lib.baselines.csg_ode.diagnostics import (
    ambiguity_ledger,
    environment_record,
    parameter_count,
)
from lib.baselines.csg_ode.model import CSGODE
from lib.baselines.csg_ode.training import run_epoch, set_reproducible_seed


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["csg_ode"], default="csg_ode")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=["interp", "extrap"], default=None)
    parser.add_argument("--obs-ratio", type=float, default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "experiments")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--n-traj-samples", type=int)
    parser.add_argument("--obsrv-std", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--normalization-mode", choices=["paper_splitwise_maxabs", "train_maxabs"])
    parser.add_argument("--kl-schedule", choices=["lgode_per_epoch", "global", "constant"])

    parser.add_argument("--adaptive-adj-dim", type=int)
    parser.add_argument("--density-activation", choices=["sigmoid", "tanh", "identity"])
    parser.add_argument("--matrix-init", "--csg-matrix-init", choices=["zero", "xavier"])
    parser.add_argument("--embedding-activation", choices=["gelu", "relu"])
    parser.add_argument("--dynamics-activation", choices=["tanh", "relu"])
    parser.add_argument("--control-init", choices=["zero", "gnn"])
    parser.add_argument("--solver", choices=["euler", "dopri5"])
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument(
        "--carry-unobserved-hidden", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--clamp-density-factor", action=argparse.BooleanOptionalAction, default=None
    )

    parser.add_argument("--ablation-aq", action="store_true")
    parser.add_argument("--ablation-no-ei", action="store_true")
    parser.add_argument("--ablation-no-g", action="store_true")
    parser.add_argument("--ablation-no-ni", action="store_true")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-validation-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--validate-communicability", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-dataset-mismatch", action="store_true")

    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "at-ode"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def resolved_config(args: argparse.Namespace):
    override_names = (
        "task",
        "obs_ratio",
        "seed",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "dropout",
        "validation_fraction",
        "n_traj_samples",
        "obsrv_std",
        "grad_clip",
        "normalization_mode",
        "kl_schedule",
        "adaptive_adj_dim",
        "density_activation",
        "matrix_init",
        "embedding_activation",
        "dynamics_activation",
        "control_init",
        "solver",
        "rtol",
        "atol",
        "carry_unobserved_hidden",
        "clamp_density_factor",
    )
    overrides = {
        name: getattr(args, name) for name in override_names if getattr(args, name) is not None
    }
    overrides.update(
        {
            "dataset": args.dataset,
            "ablation_aq": args.ablation_aq,
            "ablation_no_ei": args.ablation_no_ei,
            "ablation_no_g": args.ablation_no_g,
            "ablation_no_ni": args.ablation_no_ni,
        }
    )
    if args.config:
        return load_config(args.config, **overrides)
    return paper_config(
        args.dataset, **{key: value for key, value in overrides.items() if key != "dataset"}
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def ensure_run_directory_is_safe(run_directory: Path, resume: Path | None) -> None:
    stateful_names = ("history.json", "best.ckpt", "last.ckpt", "result.json")
    existing = [run_directory / name for name in stateful_names if (run_directory / name).exists()]
    if not existing:
        return
    if resume is None:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Run directory {run_directory} already contains {names}. Use --resume with its "
            "checkpoint or choose a new --run-name; refusing to mix training histories."
        )
    if resume.expanduser().resolve().parent != run_directory.resolve():
        raise FileExistsError(
            f"Run directory {run_directory} already has state, but --resume points to {resume}. "
            "Resume in the checkpoint's run directory or choose a new empty --run-name."
        )


def loader_generator_state(loader: Any) -> torch.Tensor | None:
    generator = getattr(loader, "generator", None)
    return generator.get_state() if generator is not None else None


def main() -> int:
    args = parse_args()
    config = resolved_config(args)
    device = resolve_device(args.device)
    set_reproducible_seed(config.seed)
    ablations = [
        name
        for name, enabled in {
            "aq": config.ablation_aq,
            "no-ei": config.ablation_no_ei,
            "no-g": config.ablation_no_g,
            "no-ni": config.ablation_no_ni,
        }.items()
        if enabled
    ]
    variant = "+".join(ablations) if ablations else "full"
    default_name = (
        f"csgode_{config.dataset}_{config.task}_{config.obs_ratio:g}_seed{config.seed}_{variant}"
    )
    run_name = args.run_name or default_name
    run_directory = args.output_root.expanduser().resolve() / run_name
    ensure_run_directory_is_safe(run_directory, args.resume)
    run_directory.mkdir(parents=True, exist_ok=True)

    cache = CommunicabilityCache(args.output_root / "_communicability_cache")
    bundle = build_datasets(config, args.data_root, communicability_cache=cache)
    bundle.audit["observation_counts"] = {
        "train": bundle.train.observation_count_summary(),
        "validation": bundle.validation.observation_count_summary(),
        "test": bundle.test.observation_count_summary(),
    }
    command = " ".join(shlex.quote(value) for value in sys.argv)
    metadata = {
        "command": command,
        "config": config.to_dict(),
        "device": str(device),
        "environment": environment_record(REPOSITORY_ROOT),
        "ambiguity_ledger": ambiguity_ledger(config),
        "dataset_audit": bundle.audit,
        "allow_dataset_mismatch": args.allow_dataset_mismatch,
        "status": "audit_only" if args.audit_only else "configured",
    }
    write_json(run_directory / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.validate_communicability:
        sample = bundle.train[0]
        validation = cache.validate(sample["original_adjacency"])
        print(f"Communicability validation: {validation}")
        write_json(run_directory / "communicability_validation.json", validation)
    if args.audit_only:
        return 0
    if not bundle.audit["paper_compatible"] and not args.allow_dataset_mismatch:
        reason = bundle.audit["feature_semantics"]["reason"]
        raise RuntimeError(
            f"Refusing a paper-comparison run for incompatible {config.dataset} data: "
            f"{reason} Supply corrected data, or pass --allow-dataset-mismatch only for "
            "a labeled plumbing/debug run."
        )
    if not bundle.audit["paper_compatible"]:
        print("WARNING: running an explicitly allowed non-paper dataset variant")

    train_loader, validation_loader, test_loader = make_loaders(
        bundle, config, num_workers=args.num_workers
    )
    model = CSGODE(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Trainable parameters: {parameter_count(model):,}")
    print(f"Batch size: {config.batch_size}")
    best_path = run_directory / "best.ckpt"
    last_path = run_directory / "last.ckpt"

    start_epoch = 1
    global_step = 0
    best_validation_mse = float("inf")
    historical_best_path: Path | None = None
    if args.resume:
        restored = load_checkpoint(args.resume, model, optimizer, device=device)
        for key in (
            "dataset",
            "task",
            "obs_ratio",
            "num_nodes",
            "input_dim",
            "latent_dim",
            "augment_dim",
            "normalization_mode",
        ):
            if restored["config"].get(key) != config.to_dict().get(key):
                raise ValueError(
                    f"Resume checkpoint {key}={restored['config'].get(key)!r} does not "
                    f"match requested {config.to_dict().get(key)!r}"
                )
        start_epoch = int(restored["epoch"]) + 1
        best_validation_mse = float(restored["best_validation_mse"])
        global_step = int(restored.get("extra", {}).get("global_step", 0))
        loader_state = restored.get("extra", {}).get("train_loader_generator_state")
        if loader_state is not None and train_loader.generator is not None:
            train_loader.generator.set_state(
                torch.as_tensor(loader_state, dtype=torch.uint8, device="cpu")
            )
        candidate = args.resume.expanduser().resolve().parent / "best.ckpt"
        historical_best_path = candidate if candidate.exists() else args.resume
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config={**config.to_dict(), "dataset_audit": bundle.audit},
        )

    history_path = run_directory / "history.json"
    history: list[dict[str, Any]] = (
        json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    )
    stale_epochs = 0
    training_start = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_summary, global_step = run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer=optimizer,
            global_step=global_step,
            limit_batches=args.limit_train_batches,
            progress=not args.no_progress,
            profile=args.profile,
            diagnostics_path=(
                run_directory / f"diagnostics_train_epoch{epoch}.npz"
                if args.save_diagnostics and epoch == start_epoch
                else None
            ),
        )
        validation_summary, _ = run_epoch(
            model,
            validation_loader,
            config,
            device,
            limit_batches=args.limit_validation_batches,
            progress=not args.no_progress,
            profile=args.profile,
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_summary,
            "validation": validation_summary,
        }
        history.append(epoch_record)
        current_mse = validation_summary["mse_lgode"]
        improved = current_mse < best_validation_mse
        if improved:
            best_validation_mse = current_mse
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                best_validation_mse=best_validation_mse,
                config=config,
                dataset_audit=bundle.audit,
                extra={
                    "global_step": global_step,
                    "validation": validation_summary,
                    "train_loader_generator_state": loader_generator_state(train_loader),
                },
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch=epoch,
            best_validation_mse=best_validation_mse,
            config=config,
            dataset_audit=bundle.audit,
            extra={
                "global_step": global_step,
                "validation": validation_summary,
                "train_loader_generator_state": loader_generator_state(train_loader),
            },
        )
        write_json(history_path, history)
        print(
            f"Epoch {epoch:04d} | train MSE {train_summary['mse_lgode']:.8f} | "
            f"val MSE {current_mse:.8f} | val x100 {validation_summary['mse_table_scaled']:.6f} | "
            f"KL {validation_summary['kl_first_point']:.6f} | best={improved}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    **{
                        f"train/{key}": value
                        for key, value in train_summary.items()
                        if isinstance(value, (int, float))
                    },
                    **{
                        f"validation/{key}": value
                        for key, value in validation_summary.items()
                        if isinstance(value, (int, float))
                    },
                    "epoch": epoch,
                },
                step=global_step,
            )
        if args.early_stopping_patience and stale_epochs >= args.early_stopping_patience:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement")
            break

    training_seconds = time.perf_counter() - training_start
    evaluation_checkpoint = best_path if best_path.exists() else historical_best_path
    if evaluation_checkpoint is None:
        raise RuntimeError("No validation-selected checkpoint is available for test evaluation")
    load_checkpoint(evaluation_checkpoint, model, device=device, restore_rng=False)
    test_summary, _ = run_epoch(
        model,
        test_loader,
        config,
        device,
        limit_batches=args.limit_test_batches,
        progress=not args.no_progress,
        profile=args.profile,
        diagnostics_path=(
            run_directory / "diagnostics_test.npz" if args.save_diagnostics else None
        ),
    )
    final = {
        **metadata,
        "status": (
            "dataset_variant_complete"
            if not bundle.audit["paper_compatible"]
            else (
                "smoke_only"
                if any(
                    value is not None
                    for value in (
                        args.limit_train_batches,
                        args.limit_validation_batches,
                        args.limit_test_batches,
                    )
                )
                else "complete"
            )
        ),
        "parameter_count": parameter_count(model),
        "best_validation_mse": best_validation_mse,
        "test": test_summary,
        "training_seconds": training_seconds,
        "best_checkpoint": str(evaluation_checkpoint),
    }
    write_json(run_directory / "result.json", final)
    print("Test summary:")
    print(json.dumps(test_summary, indent=2, sort_keys=True))
    if wandb_run is not None:
        wandb_run.log(
            {
                f"test/{key}": value
                for key, value in test_summary.items()
                if isinstance(value, (int, float))
            }
        )
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

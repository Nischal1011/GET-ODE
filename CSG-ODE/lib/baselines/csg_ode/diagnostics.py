"""Reproduction metadata, gradient summaries, and debug-bundle serialization."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import CSGODEConfig


def ambiguity_ledger(config: CSGODEConfig) -> dict[str, Any]:
    return {
        "gcrnn_gates": (
            "Canonical AGCRN-style reset/update gates with separate node-adaptive gate "
            "and candidate pools."
        ),
        "density_activation": config.density_activation,
        "adaptive_softmax_axis": "row-wise, dim=-1",
        "W1_W2_initialization": config.matrix_init,
        "unobserved_hidden_state": (
            "carry previous state" if config.carry_unobserved_hidden else "normal graph-GRU update"
        ),
        "control_initial_state": config.control_init,
        "csode_depth_convention": (
            "ControlSynth-style depth=1 means Linear(d,width), activation, Linear(width,d); "
            "the equation's shared activation is also applied to the subnetwork output."
        ),
        "augmentation": (
            "zeros are appended to z0; c has the same augmented dimension; only the first "
            f"{config.latent_dim} latent channels are decoded"
        ),
        "likelihood": (
            f"LG-ODE masked Gaussian reconstruction with fixed std={config.obsrv_std}; "
            f"KL schedule={config.kl_schedule}"
        ),
        "mse_mask": (
            "mse_lgode uses task target_mask; all-point and unobserved-only MSE are also " "logged"
        ),
        "normalization_scope": config.normalization_mode,
        "charged_graph": (
            "all signed entries are active; -1/+1 are separate relations; source diagonal "
            "is retained"
        ),
        "springs_relations": (
            "binary adjacency is used for communicability; all off-diagonal pairs carry "
            "no-spring/spring NRI relation types"
        ),
        "motion_graph": (
            "29 joints, 56 adjacency entries; six loc pose channels are used and vel "
            "derivatives are not concatenated"
        ),
        "pems_table_discrepancies": {
            "interp_0.6": {"primary": 0.2827, "ablation_table": 0.2602},
            "extrap_0.4": {"primary": 1.6607, "ablation_table": 1.7726},
        },
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def gradient_norms(model: nn.Module) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        module = name.split(".", 1)[0]
        squared = float(parameter.grad.detach().norm().cpu()) ** 2
        grouped[module] = grouped.get(module, 0.0) + squared
    return {name: value**0.5 for name, value in grouped.items()}


def environment_record(repository_root: str | Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }


def _collect_tensors(prefix: str, value: Any, destination: dict[str, np.ndarray]) -> None:
    if torch.is_tensor(value):
        destination[prefix] = value.detach().to(device="cpu").numpy()
    elif isinstance(value, dict):
        for key, child in value.items():
            _collect_tensors(f"{prefix}.{key}" if prefix else str(key), child, destination)
    elif isinstance(value, (list, tuple)) and all(torch.is_tensor(item) for item in value):
        if value:
            destination[prefix] = torch.stack(
                [item.detach().to(device="cpu") for item in value]
            ).numpy()


def save_debug_bundle(
    path: str | Path,
    batch: dict[str, Any],
    results: dict[str, Any],
    gradients: dict[str, float] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for key in (
        "observed_values",
        "observed_mask",
        "encoder_times",
        "target_values",
        "target_mask",
        "original_adjacency",
        "edge_type",
    ):
        _collect_tensors(f"batch.{key}", batch.get(key), tensors)
    for key in (
        "encoded",
        "ode_diagnostics",
        "posterior_mean",
        "posterior_std",
        "latent_trajectory",
        "predictions",
    ):
        _collect_tensors(f"result.{key}", results.get(key), tensors)
    np.savez_compressed(destination, **tensors)
    if gradients is not None:
        destination.with_suffix(".gradients.json").write_text(
            json.dumps(gradients, indent=2, sort_keys=True), encoding="utf-8"
        )
    return destination

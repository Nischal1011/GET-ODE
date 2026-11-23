"""Atomic, metadata-complete CSG-ODE checkpoint handling."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import CSGODEConfig
from .diagnostics import ambiguity_ledger


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_validation_mse: float,
    config: CSGODEConfig,
    dataset_audit: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "epoch": int(epoch),
        "best_validation_mse": float(best_validation_mse),
        "config": config.to_dict(),
        "ambiguity_ledger": ambiguity_ledger(config),
        "dataset_audit": dataset_audit,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def _byte_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    # This loader is intended for checkpoints produced by this repository.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    checkpoint_ddp = any(key.startswith("module.") for key in state)
    model_ddp = hasattr(model, "module")
    if checkpoint_ddp and not model_ddp:
        state = {key.removeprefix("module."): value for key, value in state.items()}
    elif model_ddp and not checkpoint_ddp:
        state = {f"module.{key}": value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if restore_rng and checkpoint.get("rng"):
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(_byte_tensor(rng["torch"]))
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            for device_index, state_value in enumerate(rng["cuda"]):
                if device_index >= torch.cuda.device_count():
                    break
                torch.cuda.set_rng_state(_byte_tensor(state_value), device=device_index)
    return checkpoint

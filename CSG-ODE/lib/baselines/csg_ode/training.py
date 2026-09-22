"""Training and evaluation loops shared by the CLI and tests."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CSGODEConfig
from .data_adapter import move_batch_to_device
from .diagnostics import gradient_norms, save_debug_bundle


LOGGED_METRICS = (
    "loss",
    "reconstruction_nll",
    "likelihood",
    "kl_first_point",
    "posterior_std_mean",
    "mse_all",
    "mse_unobserved",
    "mse_lgode",
    "mse_table_scaled",
)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kl_coefficient(config: CSGODEConfig, batch_index: int, global_step: int) -> float:
    if config.kl_schedule == "constant":
        return 1.0
    schedule_step = batch_index if config.kl_schedule == "lgode_per_epoch" else global_step
    if schedule_step < config.kl_wait_batches:
        return 0.0
    return 1.0 - config.kl_decay ** (schedule_step - config.kl_wait_batches)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: CSGODEConfig,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    global_step: int = 0,
    limit_batches: int | None = None,
    progress: bool = True,
    profile: bool = False,
    diagnostics_path: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in LOGGED_METRICS}
    examples = 0
    data_seconds = 0.0
    backward_seconds = 0.0
    component_seconds: dict[str, float] = {}
    batches = 0
    last_kl_coef = 1.0
    iterator = iter(loader)
    expected = len(loader) if limit_batches is None else min(len(loader), limit_batches)
    progress_bar = tqdm(range(expected), disable=not progress, leave=False)

    if profile and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index in progress_bar:
        data_start = time.perf_counter()
        batch = next(iterator)
        data_seconds += time.perf_counter() - data_start
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["sample_index"].shape[0])
        last_kl_coef = kl_coefficient(config, batch_index, global_step) if training else 1.0
        capture_diagnostics = diagnostics_path is not None and batch_index == 0

        if training:
            optimizer.zero_grad(set_to_none=True)
            results = model.compute_all_losses(
                batch,
                kl_coef=last_kl_coef,
                return_diagnostics=capture_diagnostics,
                profile=profile,
            )
            if profile:
                _synchronize(device)
                backward_start = time.perf_counter()
            results["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if profile:
                _synchronize(device)
                backward_seconds += time.perf_counter() - backward_start
            global_step += 1
        else:
            with torch.no_grad():
                results = model.compute_all_losses(
                    batch,
                    kl_coef=last_kl_coef,
                    return_diagnostics=capture_diagnostics,
                    profile=profile,
                )

        for key in LOGGED_METRICS:
            totals[key] += float(results[key].detach().cpu()) * batch_size
        for key, value in results.get("timings", {}).items():
            component_seconds[key] = component_seconds.get(key, 0.0) + float(value)
        examples += batch_size
        batches += 1
        progress_bar.set_postfix(mse=f"{float(results['mse_lgode'].detach()):.5f}")

        if capture_diagnostics:
            save_debug_bundle(
                diagnostics_path,
                batch,
                results,
                gradient_norms(model) if training else None,
            )

    if examples == 0:
        raise RuntimeError("No batches were evaluated")
    summary: dict[str, Any] = {key: value / examples for key, value in totals.items()}
    summary["examples"] = examples
    summary["batches"] = batches
    summary["kl_coef"] = last_kl_coef
    summary["profile"] = {
        "data_seconds": data_seconds,
        "backward_seconds": backward_seconds,
        **component_seconds,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if profile and device.type == "cuda"
            else None
        ),
    }
    return summary, global_step

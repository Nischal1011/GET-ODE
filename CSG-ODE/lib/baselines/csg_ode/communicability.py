"""Equation (2)-(3): finite-difference communicability edge importance."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def adjacency_hash(adjacency: Tensor | np.ndarray) -> str:
    """Return a stable hash for a topology, independent of source dtype."""

    array = (
        adjacency.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        if torch.is_tensor(adjacency)
        else np.ascontiguousarray(adjacency, dtype=np.float64)
    )
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def finite_difference_frechet(adjacency: Tensor) -> tuple[Tensor, Tensor]:
    """Compute the paper's finite-difference Frechet derivative and D in float64."""

    graph = adjacency.detach().to(device="cpu", dtype=torch.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square [N, N]")
    n_nodes = graph.shape[0]
    beta = (2.0 / float(n_nodes)) * 1e-4
    ones = torch.ones((n_nodes, 1), dtype=torch.float64)
    direction = ones @ ones.T
    identity = torch.eye(n_nodes, dtype=torch.float64)

    def exp0(matrix: Tensor) -> Tensor:
        return torch.matrix_exp(matrix) - identity

    graph_t = graph.T
    derivative = (exp0(graph_t + beta * direction) - exp0(graph_t - beta * direction)) / (
        2.0 * beta
    )
    norm = torch.linalg.matrix_norm(derivative, ord="fro")
    denominator = torch.where(
        norm == 0,
        norm.new_tensor(torch.finfo(torch.float64).eps),
        norm,
    )
    importance = graph * derivative / denominator
    return derivative, importance


def scipy_frechet(adjacency: Tensor | np.ndarray) -> Tensor:
    """Reference derivative used only by validation and tests."""

    from scipy.linalg import expm_frechet

    array = (
        adjacency.detach().to(device="cpu", dtype=torch.float64).numpy()
        if torch.is_tensor(adjacency)
        else np.asarray(adjacency, dtype=np.float64)
    )
    direction = np.ones_like(array, dtype=np.float64)
    derivative = expm_frechet(array.T, direction, compute_expm=False)
    return torch.from_numpy(np.asarray(derivative, dtype=np.float64))


class CommunicabilityCache:
    """Memory cache, with optional persistent storage, keyed by graph contents."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory: dict[str, Tensor] = {}
        self.hits = 0
        self.misses = 0
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        self._memory.clear()
        self.hits = 0
        self.misses = 0

    def get(
        self,
        adjacency: Tensor | np.ndarray,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        key = adjacency_hash(adjacency)
        if key in self._memory:
            self.hits += 1
            return self._memory[key].to(device=device, dtype=dtype)

        cache_path = self.cache_dir / f"{key}.pt" if self.cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            importance = torch.load(cache_path, map_location="cpu", weights_only=True)
            importance = importance.to(dtype=torch.float64, device="cpu")
            self.hits += 1
        else:
            _, importance = finite_difference_frechet(torch.as_tensor(adjacency))
            self.misses += 1
            if cache_path is not None:
                temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(importance, temporary)
                os.replace(temporary, cache_path)

        self._memory[key] = importance
        return importance.to(device=device, dtype=dtype)

    def get_batch(self, adjacency: Tensor) -> Tensor:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        return torch.stack(
            [
                self.get(graph, dtype=adjacency.dtype, device=adjacency.device)
                for graph in adjacency
            ],
            dim=0,
        )

    def validate(self, adjacency: Tensor | np.ndarray) -> dict[str, float]:
        graph = torch.as_tensor(adjacency, dtype=torch.float64)
        finite_difference, _ = finite_difference_frechet(graph)
        reference = scipy_frechet(graph)
        error = finite_difference - reference
        return {
            "absolute_max_error": float(error.abs().max()),
            "relative_frobenius_error": float(
                torch.linalg.matrix_norm(error, ord="fro")
                / torch.linalg.matrix_norm(reference, ord="fro").clamp_min(1e-15)
            ),
        }


DEFAULT_COMMUNICABILITY_CACHE = CommunicabilityCache()

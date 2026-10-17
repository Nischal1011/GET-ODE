from __future__ import annotations

import torch

from lib.baselines.csg_ode.communicability import (
    CommunicabilityCache,
    finite_difference_frechet,
    scipy_frechet,
)


def sample_graph() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )


def test_information_importance_shape_finiteness_and_support() -> None:
    graph = sample_graph()
    _, importance = finite_difference_frechet(graph)
    assert importance.shape == graph.shape
    assert torch.isfinite(importance).all()
    assert torch.equal(importance[graph == 0], torch.zeros_like(importance[graph == 0]))


def test_frobenius_normalization_matches_equation() -> None:
    graph = sample_graph().double()
    derivative, importance = finite_difference_frechet(graph)
    norm = torch.linalg.matrix_norm(derivative, ord="fro")
    torch.testing.assert_close(importance * norm, graph * derivative)


def test_cache_and_uncached_results_agree(tmp_path) -> None:
    graph = sample_graph()
    cache = CommunicabilityCache(tmp_path)
    first = cache.get(graph)
    second = cache.get(graph.clone())
    assert cache.misses == 1
    assert cache.hits == 1
    torch.testing.assert_close(first, second)

    reloaded = CommunicabilityCache(tmp_path)
    third = reloaded.get(graph)
    assert reloaded.hits == 1
    torch.testing.assert_close(first, third)


def test_finite_difference_matches_scipy_frechet() -> None:
    graph = sample_graph().double()
    derivative, _ = finite_difference_frechet(graph)
    reference = scipy_frechet(graph)
    torch.testing.assert_close(derivative, reference, rtol=2e-7, atol=2e-7)

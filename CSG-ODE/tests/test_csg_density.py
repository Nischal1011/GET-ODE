from __future__ import annotations

import torch

from lib.baselines.csg_ode.adaptive_gcrnn import (
    NodeAdaptiveGraphConv,
    StaticAdaptiveAdjacency,
)
from lib.baselines.csg_ode.density_adjustment import (
    DensityAdjustedGraph,
    sampling_intervals,
)


def test_appendix_a_sampling_interval_cases() -> None:
    times = torch.tensor([[0.0, 0.2, 0.6, 1.0]])
    observed = torch.tensor([[[True, False], [True, True], [False, False], [True, False]]])
    intervals = sampling_intervals(times, observed)
    # Node 0: first/only-following, interior/both, and last/only-preceding.
    torch.testing.assert_close(intervals[0, 0, 0], torch.tensor(0.2))
    torch.testing.assert_close(intervals[0, 1, 0], torch.tensor(0.5))
    torch.testing.assert_close(intervals[0, 3, 0], torch.tensor(0.8))
    # Node 1: one observation gets half of the complete observation span.
    torch.testing.assert_close(intervals[0, 1, 1], torch.tensor(0.5))
    assert intervals[0, 2].eq(0).all()


def test_pair_mask_and_unobserved_edges() -> None:
    module = DensityAdjustedGraph(3, matrix_init="zero")
    graph_mix = torch.ones(1, 3, 3)
    node_mask = torch.tensor([[1.0, 0.0, 1.0]])
    graph, _, pair_mask = module(graph_mix, node_mask, torch.zeros(1, 3))
    expected = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]])
    torch.testing.assert_close(pair_mask, expected)
    torch.testing.assert_close(graph, expected)


def test_alpha_or_zero_w2_removes_density_adjustment() -> None:
    graph_mix = torch.rand(2, 3, 3)
    node_mask = torch.ones(2, 3)
    intervals = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
    zero_alpha = DensityAdjustedGraph(3, alpha=0.0, matrix_init="xavier")
    graph, _, _ = zero_alpha(graph_mix, node_mask, intervals)
    torch.testing.assert_close(graph, graph_mix)

    zero_weights = DensityAdjustedGraph(3, alpha=0.5, matrix_init="zero")
    graph, _, _ = zero_weights(graph_mix, node_mask, intervals)
    torch.testing.assert_close(graph, graph_mix)


def test_adaptive_graph_is_row_stochastic_and_differentiable() -> None:
    adaptive = StaticAdaptiveAdjacency(4, 3)
    with torch.no_grad():
        adaptive.source_embedding.fill_(0.5)
        adaptive.target_embedding.copy_(torch.arange(12).reshape(4, 3) / 10)
    graph = adaptive()
    assert graph.ge(0).all()
    torch.testing.assert_close(graph.sum(dim=-1), torch.ones(4))
    graph.square().sum().backward()
    assert adaptive.source_embedding.grad is not None
    assert adaptive.target_embedding.grad is not None
    assert adaptive.source_embedding.grad.abs().sum() > 0
    assert adaptive.target_embedding.grad.abs().sum() > 0


def test_node_adaptive_convolution_matches_loop_and_reaches_parameters() -> None:
    torch.manual_seed(2)
    convolution = NodeAdaptiveGraphConv(3, 2, 4)
    values = torch.randn(2, 3, 3)
    graph = torch.softmax(torch.randn(2, 3, 3), dim=-1)
    representation = torch.randn(2, 3, 4, requires_grad=True)
    output = convolution(values, graph, representation)
    weights, bias = convolution.node_parameters(representation)
    assert weights.shape == (2, 3, 3, 2)

    expected = torch.empty_like(output)
    for batch_index in range(2):
        for node_index in range(3):
            aggregate = sum(
                graph[batch_index, node_index, neighbor] * values[batch_index, neighbor]
                for neighbor in range(3)
            )
            expected[batch_index, node_index] = (
                aggregate @ weights[batch_index, node_index] + bias[batch_index, node_index]
            )
    torch.testing.assert_close(output, expected)

    changed = convolution(values, graph, representation.detach() + 0.5)
    assert not torch.allclose(output.detach(), changed)
    output.sum().backward()
    assert representation.grad is not None and representation.grad.abs().sum() > 0
    assert convolution.weight_pool.grad is not None
    assert convolution.bias_pool.grad is not None

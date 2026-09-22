from __future__ import annotations

import torch

from lib.baselines.csg_ode.config import paper_config
from lib.baselines.csg_ode.csode_func import ControlSynthGraphODEFunc
from lib.baselines.csg_ode.solver import solve_joint_ode


def dynamics_config(**overrides):
    values = {
        "num_nodes": 3,
        "input_dim": 2,
        "latent_dim": 4,
        "augment_dim": 0,
        "num_relations": 2,
        "subnetwork_width": 8,
        "dropout": 0.0,
    }
    values.update(overrides)
    return paper_config("charged", **values)


def graph_inputs():
    adjacency = torch.ones(2, 3, 3)
    edge_type = torch.tensor(
        [
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        ]
    )
    return adjacency, edge_type


def test_joint_derivative_shape_and_ablation_terms() -> None:
    adjacency, edge_type = graph_inputs()
    state = torch.randn(2, 3, 8)
    function = ControlSynthGraphODEFunc(dynamics_config())
    derivative = function(torch.tensor(0.0), state, adjacency, edge_type)
    assert derivative.shape == state.shape

    no_ni = ControlSynthGraphODEFunc(dynamics_config(ablation_no_ni=True))
    assert no_ni.derivative_terms(state, adjacency, edge_type)["nonlinear"].eq(0).all()
    no_g = ControlSynthGraphODEFunc(dynamics_config(ablation_no_g=True))
    assert no_g.derivative_terms(state, adjacency, edge_type)["control"].eq(0).all()


def test_zero_control_initialization_is_deterministic() -> None:
    adjacency, edge_type = graph_inputs()
    z0 = torch.randn(2, 3, 4)
    function = ControlSynthGraphODEFunc(dynamics_config(control_init="zero"))
    first = function.initial_control(z0, adjacency, edge_type)
    second = function.initial_control(z0 * 2, adjacency, edge_type)
    assert first.eq(0).all()
    torch.testing.assert_close(first, second)


def test_euler_returns_requested_times_without_nans_and_all_modules_receive_gradients() -> None:
    torch.manual_seed(3)
    config = dynamics_config()
    function = ControlSynthGraphODEFunc(config)
    adjacency, edge_type = graph_inputs()
    z0 = torch.randn(2, 3, 4)
    c0 = function.initial_control(z0, adjacency, edge_type)
    y0 = torch.cat([z0, c0], dim=-1)
    times = torch.tensor([[0.0, 0.25, 0.6, 1.0], [0.0, 0.2, 0.7, 1.0]])
    valid = torch.ones_like(times, dtype=torch.bool)
    trajectory = solve_joint_ode(function, y0, times, valid, adjacency, edge_type, method="euler")
    assert trajectory.shape == (2, 4, 3, 8)
    assert torch.isfinite(trajectory).all()
    trajectory.square().mean().backward()

    assert function.A0.grad is not None and function.A0.grad.abs().sum() > 0
    assert function.A.grad is not None and function.A.grad.abs().sum() > 0
    assert function.control_function.weight.grad is not None
    for network in function.subnetworks:
        assert any(parameter.grad is not None for parameter in network.parameters())
    for network in function.control_gnn.message_networks:
        assert any(parameter.grad is not None for parameter in network.parameters())
    assert any(
        parameter.grad is not None for parameter in function.control_gnn.control_output.parameters()
    )

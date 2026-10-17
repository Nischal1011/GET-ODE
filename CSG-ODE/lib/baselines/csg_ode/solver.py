"""Joint z/c ODE solvers with Euler as the paper default."""

from __future__ import annotations

import torch
from torch import Tensor

from .csode_func import ControlSynthGraphODEFunc


def euler_integrate(
    function: ControlSynthGraphODEFunc,
    y0: Tensor,
    times: Tensor,
    time_valid_mask: Tensor,
    adjacency: Tensor,
    edge_type: Tensor,
) -> Tensor:
    """Integrate at each requested time, supporting padded per-sample grids."""

    if times.ndim == 1:
        times = times.unsqueeze(0).expand(y0.shape[0], -1)
    valid = time_valid_mask.bool()
    batch_size, num_times = times.shape
    state = y0
    current_time = times.new_zeros(batch_size)
    trajectory = []
    for time_index in range(num_times):
        requested = times[:, time_index]
        step_valid = valid[:, time_index]
        dt = torch.where(step_valid, requested - current_time, torch.zeros_like(requested))
        derivative = function(current_time, state, adjacency, edge_type)
        candidate = state + dt[:, None, None] * derivative
        state = torch.where(step_valid[:, None, None], candidate, state)
        current_time = torch.where(step_valid, requested, current_time)
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)


def dopri5_integrate(
    function: ControlSynthGraphODEFunc,
    y0: Tensor,
    times: Tensor,
    time_valid_mask: Tensor,
    adjacency: Tensor,
    edge_type: Tensor,
    *,
    rtol: float,
    atol: float,
) -> Tensor:
    """Integrate each sample separately because padded query grids may differ."""

    from torchdiffeq import odeint

    outputs: list[Tensor] = []
    max_times = times.shape[1]
    for batch_index in range(y0.shape[0]):
        query = times[batch_index, time_valid_mask[batch_index].bool()]
        if query.numel() == 0:
            outputs.append(y0.new_zeros(max_times, *y0.shape[1:]))
            continue
        prepend_zero = not torch.isclose(query[0], query.new_zeros(())).item()
        integration_times = torch.cat([query.new_zeros(1), query]) if prepend_zero else query

        def wrapped(time: Tensor, state: Tensor) -> Tensor:
            return function(
                time,
                state,
                adjacency[batch_index : batch_index + 1],
                edge_type[batch_index : batch_index + 1],
            )

        solved = odeint(
            wrapped,
            y0[batch_index : batch_index + 1],
            integration_times,
            method="dopri5",
            rtol=rtol,
            atol=atol,
        )
        if prepend_zero:
            solved = solved[1:]
        solved = solved[:, 0]
        padded = y0.new_zeros(max_times, *y0.shape[1:])
        padded[: solved.shape[0]] = solved
        if solved.shape[0] < max_times:
            padded[solved.shape[0] :] = solved[-1]
        outputs.append(padded)
    return torch.stack(outputs, dim=0)


def solve_joint_ode(
    function: ControlSynthGraphODEFunc,
    y0: Tensor,
    times: Tensor,
    time_valid_mask: Tensor,
    adjacency: Tensor,
    edge_type: Tensor,
    *,
    method: str = "euler",
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Tensor:
    if method == "euler":
        return euler_integrate(function, y0, times, time_valid_mask, adjacency, edge_type)
    if method == "dopri5":
        return dopri5_integrate(
            function,
            y0,
            times,
            time_valid_mask,
            adjacency,
            edge_type,
            rtol=rtol,
            atol=atol,
        )
    raise ValueError(f"Unsupported ODE solver: {method}")

"""Appendix-A sampling intervals and Equations (5)-(6)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def sampling_intervals(
    encoder_times: Tensor,
    observed_mask: Tensor,
    time_valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute R only at observed node-time entries; unobserved entries stay zero."""

    if encoder_times.ndim == 1:
        encoder_times = encoder_times.unsqueeze(0)
    if observed_mask.ndim == 2:
        observed_mask = observed_mask.unsqueeze(0)
    if observed_mask.ndim != 3:
        raise ValueError("observed_mask must have shape [B, T, N]")
    if encoder_times.shape[:2] != observed_mask.shape[:2]:
        raise ValueError("encoder_times and observed_mask must share [B, T]")

    batch_size, _, num_nodes = observed_mask.shape
    if time_valid_mask is None:
        time_valid_mask = torch.ones_like(encoder_times, dtype=torch.bool)
    else:
        time_valid_mask = time_valid_mask.to(dtype=torch.bool)

    intervals = torch.zeros_like(observed_mask, dtype=encoder_times.dtype)
    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(time_valid_mask[batch_index], as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            continue
        valid_times = encoder_times[batch_index, valid_indices]
        span = valid_times[-1] - valid_times[0]
        for node_index in range(num_nodes):
            node_observed = observed_mask[batch_index, valid_indices, node_index].bool()
            positions = torch.nonzero(node_observed, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            if positions.numel() == 1:
                intervals[batch_index, valid_indices[positions[0]], node_index] = span / 2.0
                continue

            node_times = valid_times[positions]
            values = torch.empty_like(node_times)
            values[0] = node_times[1] - node_times[0]
            values[-1] = node_times[-1] - node_times[-2]
            if positions.numel() > 2:
                values[1:-1] = (node_times[2:] - node_times[:-2]) / 2.0
            intervals[batch_index, valid_indices[positions], node_index] = values
    return intervals


def activate_density(intervals: Tensor, activation: str) -> Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(intervals)
    if activation == "tanh":
        return torch.tanh(intervals)
    if activation == "identity":
        return intervals
    raise ValueError(f"Unsupported density activation: {activation}")


class DensityAdjustedGraph(nn.Module):
    """Construct one streamed density-aware graph snapshot."""

    def __init__(
        self,
        num_nodes: int,
        *,
        alpha: float = 0.5,
        activation: str = "sigmoid",
        clamp_factor: bool = False,
        matrix_init: str = "zero",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.activation = activation
        self.clamp_factor = clamp_factor
        self.W2 = nn.Parameter(torch.empty(num_nodes, num_nodes))
        if matrix_init == "zero":
            nn.init.zeros_(self.W2)
        elif matrix_init == "xavier":
            nn.init.xavier_uniform_(self.W2)
        else:
            raise ValueError("matrix_init must be zero or xavier")

    def forward(
        self,
        graph_mix: Tensor,
        node_observed: Tensor,
        intervals: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if graph_mix.ndim == 2:
            graph_mix = graph_mix.unsqueeze(0)
        node_observed = node_observed.to(dtype=graph_mix.dtype)
        pair_mask = node_observed.unsqueeze(-1) * node_observed.unsqueeze(-2)
        activated = activate_density(intervals, self.activation)
        difference = (activated.unsqueeze(-1) - activated.unsqueeze(-2)).abs()
        factor = 1.0 - self.alpha * self.W2.unsqueeze(0) * difference
        if self.clamp_factor:
            factor = factor.clamp_min(0.0)
        snapshot = graph_mix * pair_mask * factor
        return snapshot, factor, pair_mask

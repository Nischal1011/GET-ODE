"""Observation-derived node-adaptive graph recurrent components."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StaticAdaptiveAdjacency(nn.Module):
    """Equation (1): row-softmax(ReLU(E_s E_t^T))."""

    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super().__init__()
        self.source_embedding = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.target_embedding = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.source_embedding)
        nn.init.xavier_uniform_(self.target_embedding)

    def forward(self) -> Tensor:
        scores = torch.relu(self.source_embedding @ self.target_embedding.T)
        return torch.softmax(scores, dim=-1)


class NodeAdaptiveGraphConv(nn.Module):
    """Equation (11): a first-order graph kernel generated from sample-specific Q."""

    def __init__(self, input_dim: int, output_dim: int, parameter_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.parameter_dim = parameter_dim
        self.weight_pool = nn.Parameter(torch.empty(parameter_dim, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.empty(parameter_dim, output_dim))
        nn.init.xavier_uniform_(self.weight_pool)
        nn.init.zeros_(self.bias_pool)

    def node_parameters(self, node_representation: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.einsum("bnq,qio->bnio", node_representation, self.weight_pool)
        bias = torch.einsum("bnq,qo->bno", node_representation, self.bias_pool)
        return weights, bias

    def forward(self, values: Tensor, graph: Tensor, node_representation: Tensor) -> Tensor:
        weights, bias = self.node_parameters(node_representation)
        aggregated = torch.bmm(graph, values)
        return torch.einsum("bni,bnio->bno", aggregated, weights) + bias


class NodeAdaptiveGraphGRUCell(nn.Module):
    """Canonical AGCRN-style graph GRU with CSG-ODE's first-order support."""

    def __init__(self, input_dim: int, hidden_dim: int, parameter_dim: int) -> None:
        super().__init__()
        combined_dim = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.gates = NodeAdaptiveGraphConv(combined_dim, 2 * hidden_dim, parameter_dim)
        self.candidate = NodeAdaptiveGraphConv(combined_dim, hidden_dim, parameter_dim)

    def forward(
        self,
        values: Tensor,
        hidden: Tensor,
        graph: Tensor,
        node_representation: Tensor,
        node_observed: Tensor | None = None,
        *,
        carry_unobserved: bool = True,
    ) -> Tensor:
        gate_input = torch.cat([values, hidden], dim=-1)
        reset, update = torch.sigmoid(self.gates(gate_input, graph, node_representation)).chunk(
            2, dim=-1
        )
        candidate_input = torch.cat([values, reset * hidden], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input, graph, node_representation))
        next_hidden = update * hidden + (1.0 - update) * candidate
        if carry_unobserved and node_observed is not None:
            mask = node_observed.to(dtype=hidden.dtype).unsqueeze(-1)
            next_hidden = mask * next_hidden + (1.0 - mask) * hidden
        return next_hidden

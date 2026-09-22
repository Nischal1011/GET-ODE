"""Equation (14): relation-aware ControlSynth Graph ODE vector field."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import CSGODEConfig
from .layers import MLP, activation_module


class RelationAwareControlGNN(nn.Module):
    """NRI-style edge messages over the verified original physical graph."""

    def __init__(self, state_dim: int, width: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.message_networks = nn.ModuleList(
            [
                MLP(
                    2 * state_dim,
                    state_dim,
                    width=width,
                    depth=1,
                    activation="tanh",
                    dropout=dropout,
                )
                for _ in range(num_relations)
            ]
        )
        self.control_output = MLP(
            2 * state_dim,
            state_dim,
            width=width,
            depth=1,
            activation="tanh",
            dropout=dropout,
        )

    def forward(self, states: Tensor, adjacency: Tensor, edge_type: Tensor) -> Tensor:
        batch_size, num_nodes, state_dim = states.shape
        if adjacency.shape != edge_type.shape:
            raise ValueError("adjacency and edge_type must share [B, N, N]")
        aggregate = states.new_zeros(batch_size, num_nodes, state_dim)
        # edge_type=-1 excludes a pair. Springs intentionally assigns relation 0
        # to off-diagonal no-spring pairs, matching the repository's NRI decoder.
        active = edge_type >= 0
        for batch_index in range(batch_size):
            for relation_index, network in enumerate(self.message_networks):
                indices = torch.nonzero(
                    active[batch_index] & (edge_type[batch_index] == relation_index),
                    as_tuple=False,
                )
                if indices.numel() == 0:
                    continue
                receivers, senders = indices[:, 0], indices[:, 1]
                edge_inputs = torch.cat(
                    [states[batch_index, receivers], states[batch_index, senders]], dim=-1
                )
                messages = network(edge_inputs)
                aggregate[batch_index].index_add_(0, receivers, messages)
        return self.control_output(torch.cat([states, aggregate], dim=-1))


class ControlSynthGraphODEFunc(nn.Module):
    """Coupled derivative for y=[z,c], with c using the augmented z dimension."""

    def __init__(self, config: CSGODEConfig) -> None:
        super().__init__()
        self.config = config
        dimension = config.dynamics_dim
        self.dimension = dimension
        self.A0 = nn.Parameter(torch.empty(dimension, dimension))
        self.A = nn.Parameter(torch.empty(config.num_subnetworks, dimension, dimension))
        nn.init.xavier_uniform_(self.A0)
        for matrix in self.A:
            nn.init.xavier_uniform_(matrix)
        self.subnetworks = nn.ModuleList(
            [
                MLP(
                    dimension,
                    dimension,
                    width=config.subnetwork_width,
                    depth=config.subnetwork_depth,
                    activation=config.dynamics_activation,
                    dropout=config.dropout,
                )
                for _ in range(config.num_subnetworks)
            ]
        )
        self.nonlinear_activation = activation_module(config.dynamics_activation)

        # Official ControlSynth examples use a single linear control function.
        self.control_function = nn.Linear(dimension, dimension)
        nn.init.xavier_uniform_(self.control_function.weight)
        nn.init.zeros_(self.control_function.bias)
        self.control_gnn = RelationAwareControlGNN(
            dimension,
            config.subnetwork_width,
            config.num_relations,
            config.dropout,
        )
        self.nfe = 0

    def initial_control(self, z0: Tensor, adjacency: Tensor, edge_type: Tensor) -> Tensor:
        if self.config.control_init == "zero":
            return torch.zeros_like(z0)
        return self.control_gnn(z0, adjacency, edge_type)

    def derivative_terms(
        self,
        y: Tensor,
        adjacency: Tensor,
        edge_type: Tensor,
    ) -> dict[str, Tensor | list[Tensor]]:
        z, control = y.split(self.dimension, dim=-1)
        linear = torch.einsum("ij,bnj->bni", self.A0, z)
        nonlinear_terms: list[Tensor] = []
        for index, network in enumerate(self.subnetworks):
            activated = self.nonlinear_activation(network(z))
            nonlinear_terms.append(torch.einsum("ij,bnj->bni", self.A[index], activated))
        nonlinear = torch.stack(nonlinear_terms, dim=0).sum(dim=0)
        if self.config.ablation_no_ni:
            nonlinear = torch.zeros_like(linear)
        control_term = self.control_function(control)
        if self.config.ablation_no_g:
            control_term = torch.zeros_like(linear)
        control_dot = self.control_gnn(z, adjacency, edge_type)
        z_dot = linear + nonlinear + control_term
        return {
            "linear": linear,
            "nonlinear_terms": nonlinear_terms,
            "nonlinear": nonlinear,
            "control": control_term,
            "control_dot": control_dot,
            "z_dot": z_dot,
        }

    def forward(
        self,
        time: Tensor,
        y: Tensor,
        adjacency: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        del time
        self.nfe += 1
        terms = self.derivative_terms(y, adjacency, edge_type)
        return torch.cat([terms["z_dot"], terms["control_dot"]], dim=-1)

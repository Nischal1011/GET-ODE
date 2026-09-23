"""Equations (1)-(13): CSG-ODE latent-distribution generator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .adaptive_gcrnn import NodeAdaptiveGraphGRUCell, StaticAdaptiveAdjacency
from .communicability import CommunicabilityCache, DEFAULT_COMMUNICABILITY_CACHE
from .config import CSGODEConfig
from .density_adjustment import DensityAdjustedGraph, sampling_intervals
from .layers import MLP


class CSGEncoder(nn.Module):
    """Infer a diagonal Gaussian initial-state posterior for every node."""

    def __init__(
        self,
        config: CSGODEConfig,
        *,
        communicability_cache: CommunicabilityCache | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = config.num_nodes
        self.adaptive_graph = StaticAdaptiveAdjacency(
            config.num_nodes, config.resolved_adaptive_adj_dim
        )
        self.W1 = nn.Parameter(torch.empty(config.num_nodes, config.num_nodes))
        if config.matrix_init == "zero":
            nn.init.zeros_(self.W1)
        else:
            nn.init.xavier_uniform_(self.W1)

        self.density_graph = DensityAdjustedGraph(
            config.num_nodes,
            alpha=config.alpha,
            activation=config.density_activation,
            clamp_factor=config.clamp_density_factor,
            matrix_init=config.matrix_init,
        )
        k = config.observation_embedding_dim
        q = config.node_parameter_dim
        h = config.hidden_dim
        self.observation_embedding = MLP(
            config.input_dim,
            k,
            width=k,
            depth=1,
            activation=config.embedding_activation,
            dropout=config.dropout,
        )
        self.node_representation = MLP(
            k,
            q,
            width=max(k, q),
            depth=1,
            activation=config.embedding_activation,
            dropout=config.dropout,
        )
        self.missing_node_embedding = nn.Parameter(torch.zeros(config.num_nodes, k))
        self.free_node_representation = nn.Parameter(torch.empty(config.num_nodes, q))
        nn.init.xavier_uniform_(self.free_node_representation)

        self.recurrent_cell = NodeAdaptiveGraphGRUCell(k, h, q)
        self.posterior = MLP(
            h,
            2 * config.latent_dim,
            width=max(h, config.latent_dim),
            depth=1,
            activation=config.embedding_activation,
            dropout=config.dropout,
        )
        self.communicability_cache = (
            communicability_cache
            if communicability_cache is not None
            else DEFAULT_COMMUNICABILITY_CACHE
        )

    def _importance(self, batch: dict[str, Tensor], adjacency: Tensor) -> Tensor:
        cached = batch.get("information_importance")
        if cached is not None:
            return cached.to(device=adjacency.device, dtype=adjacency.dtype)
        return self.communicability_cache.get_batch(adjacency)

    def _mean_embedding(
        self,
        embeddings: Tensor,
        observed_mask: Tensor,
        time_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        valid_observed = observed_mask.bool() & time_valid_mask.bool().unsqueeze(-1)
        weights = valid_observed.to(dtype=embeddings.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1)
        mean = (embeddings * weights).sum(dim=1) / counts.clamp_min(1.0)
        missing = counts.squeeze(-1) == 0
        missing_values = self.missing_node_embedding.unsqueeze(0).expand_as(mean)
        mean = torch.where(missing.unsqueeze(-1), missing_values, mean)
        return mean, counts.squeeze(-1)

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        values = batch["observed_values"]
        node_mask = batch["observed_mask"].bool()
        feature_mask = batch["feature_mask"].to(dtype=values.dtype)
        times = batch["encoder_times"]
        time_valid = batch["time_valid_mask"].bool()
        adjacency = batch["original_adjacency"].to(dtype=values.dtype)

        batch_size, num_times, num_nodes, _ = values.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

        embeddings = self.observation_embedding(values * feature_mask)
        embeddings = embeddings * node_mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        mean_embedding, observation_counts = self._mean_embedding(embeddings, node_mask, time_valid)
        if self.config.ablation_aq:
            node_representation = self.free_node_representation.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        else:
            node_representation = self.node_representation(mean_embedding)

        adaptive = self.adaptive_graph()
        adaptive_batch = adaptive.unsqueeze(0).expand(batch_size, -1, -1)
        importance = self._importance(batch, adjacency)
        graph_mix = adaptive_batch
        if not self.config.ablation_no_ei:
            graph_mix = graph_mix + self.W1.unsqueeze(0) * importance

        intervals = sampling_intervals(times, node_mask, time_valid)
        hidden = values.new_zeros(batch_size, num_nodes, self.config.hidden_dim)
        snapshots: dict[str, Tensor] = {}
        selected = {0, max(0, num_times // 2), max(0, num_times - 1)}
        selected_factors: dict[str, Tensor] = {}
        for time_index in range(num_times):
            current_observed = node_mask[:, time_index] & time_valid[:, time_index].unsqueeze(-1)
            graph, factor, _ = self.density_graph(
                graph_mix,
                current_observed,
                intervals[:, time_index],
            )
            hidden = self.recurrent_cell(
                embeddings[:, time_index],
                hidden,
                graph,
                node_representation,
                current_observed,
                carry_unobserved=self.config.carry_unobserved_hidden,
            )
            if return_diagnostics and time_index in selected:
                snapshots[str(time_index)] = graph.detach()
                selected_factors[str(time_index)] = factor.detach()

        posterior_params = self.posterior(hidden)
        mean, raw_scale = posterior_params.chunk(2, dim=-1)
        std = F.softplus(raw_scale) + 1e-6
        kl_per_dimension = 0.5 * (mean.square() + std.square() - 1.0 - 2.0 * std.log())
        result: dict[str, Tensor | dict[str, Tensor]] = {
            "mean": mean,
            "std": std,
            "kl_per_node": kl_per_dimension.sum(dim=-1),
            "kl": kl_per_dimension.mean(),
            "adaptive_graph": adaptive,
            "importance": importance,
            "graph_mix": graph_mix,
            "mean_embedding": mean_embedding,
            "node_representation": node_representation,
            "observation_counts": observation_counts,
            "intervals": intervals,
        }
        if return_diagnostics:
            result["graph_snapshots"] = snapshots
            result["density_factors"] = selected_factors
            result["edge_importance_term"] = self.W1.unsqueeze(0) * importance
        return result

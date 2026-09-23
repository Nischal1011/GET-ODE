import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.edge_latents import TimeGridInterpolator, gumbel_softmax_sample


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def categorical_prior_probs(
    edge_labels,
    batch_size: int,
    num_edges: int,
    edge_types: int,
    mode: str,
    label_probability: float,
    device,
):
    """Build one categorical prior per ordered physical edge."""
    if mode == "uniform" or edge_labels is None:
        return torch.full(
            (batch_size, num_edges, edge_types),
            1.0 / edge_types,
            device=device,
            dtype=torch.float32,
        )

    if mode != "graph":
        raise ValueError(f"Unknown edge prior mode: {mode}")
    if edge_types < 2:
        raise ValueError("Graph priors require at least two relation types.")
    if not 0.0 <= label_probability <= 1.0:
        raise ValueError("edge_prior_p must be in [0, 1].")

    labels = edge_labels.to(device=device, dtype=torch.long).reshape(batch_size, num_edges)
    if torch.any(labels < 0) or torch.any(labels >= edge_types):
        raise ValueError("Physical edge labels fall outside the configured relation types.")

    other_probability = (1.0 - label_probability) / (edge_types - 1)
    probs = torch.full(
        (batch_size, num_edges, edge_types),
        other_probability,
        device=device,
        dtype=torch.float32,
    )
    return probs.scatter(-1, labels.unsqueeze(-1), label_probability)


class RelationalEvidenceHead(nn.Module):
    """Infer nonnegative, unnormalized relation evidence on temporal event edges."""

    def __init__(
        self,
        event_dim: int,
        edge_types: int,
        hidden_dim: int = 64,
        init_bias: float = -2.0,
    ):
        super().__init__()
        input_dim = 2 * event_dim + 3
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, edge_types)
        nn.init.constant_(self.output.bias, init_bias)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        delta_t: torch.Tensor,
        mask_src: torch.Tensor,
        mask_dst: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = delta_t.reshape(-1, 1).to(dtype=h_src.dtype)
        mask_src = mask_src.reshape(-1, 1).to(dtype=h_src.dtype)
        mask_dst = mask_dst.reshape(-1, 1).to(dtype=h_src.dtype)
        evidence_input = torch.cat(
            [h_dst, h_src, delta_t, mask_dst, mask_src], dim=-1
        )
        return F.softplus(self.output(F.gelu(self.hidden(evidence_input))))

    def gradient_norm(self) -> torch.Tensor:
        grads = [
            parameter.grad.detach().float().norm()
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if not grads:
            return self.output.weight.new_zeros((), dtype=torch.float32)
        return torch.stack(grads).norm()


class _PhysicalEdgeOrdering(nn.Module):
    """Canonical sender/receiver ordering derived from the generator matrices."""

    def __init__(self, n_atoms: int, rel_send: torch.Tensor, rel_rec: torch.Tensor):
        super().__init__()
        if rel_send.ndim != 2 or rel_rec.ndim != 2 or rel_send.shape != rel_rec.shape:
            raise ValueError("rel_send and rel_rec must have matching [E, N] shapes.")
        if rel_send.size(1) != n_atoms:
            raise ValueError("Generator relation matrices do not match n_atoms.")

        edge_sender = rel_send.argmax(dim=-1).to(dtype=torch.long)
        edge_receiver = rel_rec.argmax(dim=-1).to(dtype=torch.long)
        if torch.any(edge_sender == edge_receiver):
            raise ValueError("The generator physical edge ordering must exclude self-pairs.")

        pair_to_edge = torch.full(
            (n_atoms, n_atoms),
            -1,
            dtype=torch.long,
            device=rel_send.device,
        )
        pair_to_edge[edge_sender, edge_receiver] = torch.arange(
            edge_sender.numel(), device=rel_send.device
        )

        self.n_atoms = n_atoms
        self.register_buffer("edge_sender", edge_sender, persistent=False)
        self.register_buffer("edge_receiver", edge_receiver, persistent=False)
        self.register_buffer("pair_to_edge", pair_to_edge, persistent=False)

    @property
    def num_physical_edges(self) -> int:
        return self.edge_sender.numel()


class StaticRelationPosterior(_PhysicalEdgeOrdering):
    """Topology-only categorical posterior used by the controlled ablation."""

    def __init__(
        self,
        n_atoms: int,
        edge_types: int,
        rel_send: torch.Tensor,
        rel_rec: torch.Tensor,
        prior_mode: str,
        edge_prior_p: float,
        prior_strength: float,
        tau_gumbel: float,
        hard_gumbel: bool,
    ):
        super().__init__(n_atoms, rel_send, rel_rec)
        self.edge_types = edge_types
        self.prior_mode = prior_mode
        self.edge_prior_p = edge_prior_p
        self.prior_strength = prior_strength
        self.tau_gumbel = tau_gumbel
        self.hard_gumbel = hard_gumbel

    def build(self, batch_en, t_grid: torch.Tensor, sample: bool = False):
        node_graph = batch_en.batch
        batch_size = int(node_graph.max().item()) + 1
        prior_probs = categorical_prior_probs(
            getattr(batch_en, "physical_edge_labels", None),
            batch_size,
            self.num_physical_edges,
            self.edge_types,
            self.prior_mode,
            self.edge_prior_p,
            node_graph.device,
        )
        q_probs_grid = prior_probs.unsqueeze(0).expand(t_grid.numel(), -1, -1, -1)
        concentration_grid = self.prior_strength * q_probs_grid
        provider = _make_provider(
            t_grid, q_probs_grid, sample, self.tau_gumbel, self.hard_gumbel
        )
        entropy = -(q_probs_grid * q_probs_grid.clamp_min(1e-8).log()).sum(-1)
        empty = prior_probs.new_empty((0, self.edge_types))
        extras = {
            "event_evidence": empty,
            "event_evidence_mean": prior_probs.new_zeros(()),
            "event_evidence_std": prior_probs.new_zeros(()),
            "event_evidence_max": prior_probs.new_zeros(()),
            "fraction_evidence_near_zero": prior_probs.new_ones(()),
            "posterior_concentration_grid": concentration_grid,
            "posterior_total_concentration": concentration_grid.sum(-1),
            "q_probs_grid": q_probs_grid,
            "prior_probs": prior_probs,
            "posterior_entropy": entropy,
            "age_kernel": prior_probs.new_empty((self.edge_types, 0)),
            "decay_rates": prior_probs.new_zeros(self.edge_types),
            "source_event_times": prior_probs.new_empty(0),
            "physical_pair_indices": torch.empty(
                (0, 3), device=node_graph.device, dtype=torch.long
            ),
        }
        return provider, extras


class EvidenceTransportPosterior(_PhysicalEdgeOrdering):
    """Factorized categorical posterior from additive, age-transported evidence.

    A cross-object temporal edge contributes evidence when its receiver event is
    observed. That receiver time is therefore the source time. Evidence enters
    lag bin zero, advances toward older bins, decays by relation type, and exits
    through the open oldest-bin boundary.
    """

    VALID_ABLATIONS = {
        "full",
        "instantaneous",
        "shuffle-times",
        "shuffle-values",
        "prior-only",
    }

    def __init__(
        self,
        n_atoms: int,
        edge_types: int,
        rel_send: torch.Tensor,
        rel_rec: torch.Tensor,
        evidence_head: RelationalEvidenceHead,
        K_lag: int = 16,
        lag_max: float = 1.0,
        transport_velocity: float = 1.0,
        decay_init: float = 1.0,
        age_kernel: str = "uniform",
        prior_mode: str = "uniform",
        edge_prior_p: float = 0.9,
        prior_strength: float = 1.0,
        tau_gumbel: float = 1.0,
        hard_gumbel: bool = False,
        ablation: str = "full",
        eps: float = 1e-8,
    ):
        super().__init__(n_atoms, rel_send, rel_rec)
        if K_lag < 2:
            raise ValueError("Evidence transport requires K_lag >= 2.")
        if lag_max <= 0.0:
            raise ValueError("lag_max must be positive.")
        if transport_velocity < 0.0:
            raise ValueError("attn_v/transport velocity must be nonnegative.")
        if prior_strength <= 0.0:
            raise ValueError("edge_prior_strength must be positive.")
        if age_kernel not in {"uniform", "exponential", "learned"}:
            raise ValueError(f"Unknown age kernel: {age_kernel}")
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Unknown evidence ablation: {ablation}")

        self.edge_types = edge_types
        self.evidence_head = evidence_head
        self.K_lag = K_lag
        self.lag_max = float(lag_max)
        self.transport_velocity = float(transport_velocity)
        self.age_kernel_mode = age_kernel
        self.prior_mode = prior_mode
        self.edge_prior_p = float(edge_prior_p)
        self.prior_strength = float(prior_strength)
        self.tau_gumbel = float(tau_gumbel)
        self.hard_gumbel = bool(hard_gumbel)
        self.ablation = ablation
        self.eps = float(eps)

        self.raw_decay_rates = nn.Parameter(
            torch.full((edge_types,), _inverse_softplus(decay_init))
        )
        lag_grid = torch.linspace(0.0, self.lag_max, K_lag)
        self.register_buffer("lag_grid", lag_grid, persistent=False)

        if age_kernel == "learned":
            self.raw_age_kernel = nn.Parameter(torch.zeros(edge_types, K_lag))
            self.register_buffer("fixed_age_kernel", torch.empty(0), persistent=False)
        else:
            self.register_parameter("raw_age_kernel", None)
            if age_kernel == "uniform":
                kernel = torch.full((edge_types, K_lag), 1.0 / K_lag)
            else:
                kernel = torch.exp(-lag_grid).expand(edge_types, -1)
                kernel = kernel / kernel.sum(dim=-1, keepdim=True)
            self.register_buffer("fixed_age_kernel", kernel, persistent=False)

    @property
    def decay_rates(self) -> torch.Tensor:
        return F.softplus(self.raw_decay_rates.float())

    def get_age_kernel(self) -> torch.Tensor:
        if self.raw_age_kernel is not None:
            return torch.softmax(self.raw_age_kernel.float(), dim=-1)
        return self.fixed_age_kernel.float()

    def _node_mask_quality(self, batch_en) -> torch.Tensor:
        if hasattr(batch_en, "mask_quality"):
            return batch_en.mask_quality.float().reshape(-1)
        if hasattr(batch_en, "node_mask"):
            mask = batch_en.node_mask.float()
            return mask.reshape(mask.size(0), -1).mean(dim=-1)
        return batch_en.x.new_ones(batch_en.x.size(0), dtype=torch.float32)

    def _map_event_edges(self, batch_en):
        if not hasattr(batch_en, "object_id"):
            raise AttributeError(
                "Evidence mode requires batch_en.object_id from transfer_one_graph()."
            )
        edge_index = batch_en.edge_index
        src, dst = edge_index[0], edge_index[1]
        node_graph = batch_en.batch
        if not torch.equal(node_graph[src], node_graph[dst]):
            raise AssertionError("A temporal edge connects nodes from different graphs.")

        src_object = batch_en.object_id[src].long()
        dst_object = batch_en.object_id[dst].long()
        same_object = src_object == dst_object
        if hasattr(batch_en, "edge_same"):
            flagged_same = batch_en.edge_same.reshape(-1) > 0.5
            if not torch.equal(same_object, flagged_same):
                raise AssertionError("edge_same disagrees with explicit object_id metadata.")

        valid_indices = (~same_object).nonzero(as_tuple=False).flatten()
        src_object = src_object[valid_indices]
        dst_object = dst_object[valid_indices]
        graph_ids = node_graph[src[valid_indices]].long()
        physical_edge_ids = self.pair_to_edge[src_object, dst_object]
        if torch.any(physical_edge_ids < 0):
            raise AssertionError("A cross-object event edge has no generator edge mapping.")

        source_times = batch_en.pos[dst[valid_indices]].float().reshape(-1)
        physical_pairs = torch.stack(
            [graph_ids, src_object, dst_object], dim=-1
        )
        return valid_indices, graph_ids, physical_edge_ids, source_times, physical_pairs

    def _advance_state(self, concentration: torch.Tensor, delta_t: float) -> torch.Tensor:
        """Conservative upwind aging with CFL substeps and open oldest-bin outflow."""
        delta_t = max(float(delta_t), 0.0)
        if delta_t == 0.0:
            return concentration

        delta_lag = self.lag_max / (self.K_lag - 1)
        courant = self.transport_velocity * delta_t / delta_lag
        num_substeps = max(1, int(math.ceil(courant)))
        sub_dt = delta_t / num_substeps
        sub_courant = self.transport_velocity * sub_dt / delta_lag
        decay = torch.exp(
            -self.decay_rates.view(1, 1, -1, 1) * sub_dt
        ).to(device=concentration.device)

        state = concentration
        for _ in range(num_substeps):
            shifted = F.pad(state[..., :-1], (1, 0))
            state = state + sub_courant * (shifted - state)
            state = state * decay
        return state

    def _summarize(self, concentration: torch.Tensor) -> torch.Tensor:
        kernel = self.get_age_kernel().to(device=concentration.device)
        return (concentration * kernel.view(1, 1, self.edge_types, self.K_lag)).sum(-1)

    def _shuffle_within_graph(self, values: torch.Tensor, graph_ids: torch.Tensor):
        shuffled = values.clone()
        for graph_id in torch.unique(graph_ids):
            indices = (graph_ids == graph_id).nonzero(as_tuple=False).flatten()
            permutation = torch.randperm(indices.numel(), device=indices.device)
            shuffled[indices] = values[indices[permutation]]
        return shuffled

    def build(self, batch_en, t_grid: torch.Tensor, sample: bool = False):
        if not hasattr(batch_en, "event_embeddings"):
            raise AttributeError(
                "Evidence mode requires final encoder event embeddings on batch_en."
            )
        if t_grid.ndim != 1 or t_grid.numel() == 0:
            raise ValueError("time_steps_to_predict must be a nonempty 1D tensor.")
        if torch.any(t_grid[1:] < t_grid[:-1]):
            raise ValueError("time_steps_to_predict must be sorted.")

        valid, graph_ids, physical_edge_ids, source_times, physical_pairs = (
            self._map_event_edges(batch_en)
        )
        edge_index = batch_en.edge_index
        src = edge_index[0, valid]
        dst = edge_index[1, valid]
        embeddings = batch_en.event_embeddings
        mask_quality = self._node_mask_quality(batch_en)

        if self.ablation == "prior-only":
            event_evidence = embeddings.new_zeros(
                (valid.numel(), self.edge_types), dtype=torch.float32
            )
        else:
            event_evidence = self.evidence_head(
                embeddings[src],
                embeddings[dst],
                batch_en.edge_attr[valid],
                mask_quality[src],
                mask_quality[dst],
            ).float()

        if self.ablation == "shuffle-times":
            source_times = self._shuffle_within_graph(source_times, graph_ids)
        elif self.ablation == "shuffle-values":
            event_evidence = self._shuffle_within_graph(event_evidence, graph_ids)

        node_graph = batch_en.batch
        batch_size = int(node_graph.max().item()) + 1
        num_edges = self.num_physical_edges
        prior_probs = categorical_prior_probs(
            getattr(batch_en, "physical_edge_labels", None),
            batch_size,
            num_edges,
            self.edge_types,
            self.prior_mode,
            self.edge_prior_p,
            embeddings.device,
        )
        prior_concentration = self.prior_strength * prior_probs

        concentration = torch.zeros(
            batch_size,
            num_edges,
            self.edge_types,
            self.K_lag,
            device=embeddings.device,
            dtype=torch.float32,
        )
        timeline = torch.unique(torch.cat([source_times, t_grid.float()]), sorted=True)
        q_slots = [None] * t_grid.numel()
        concentration_slots = [None] * t_grid.numel()
        previous_time = float(timeline[0].item())

        for current_time_tensor in timeline:
            current_time = float(current_time_tensor.item())
            if self.ablation != "instantaneous":
                concentration = self._advance_state(
                    concentration, current_time - previous_time
                )

            source_mask = source_times == current_time_tensor
            if torch.any(source_mask):
                flat_slots = (
                    graph_ids[source_mask] * num_edges
                    + physical_edge_ids[source_mask]
                )
                injected = concentration.new_zeros(
                    batch_size * num_edges, self.edge_types
                ).index_add(0, flat_slots, event_evidence[source_mask])
                injected = injected.reshape(
                    batch_size, num_edges, self.edge_types, 1
                )
                concentration = concentration + F.pad(
                    injected, (0, self.K_lag - 1)
                )

            query_indices = (t_grid.float() == current_time_tensor).nonzero(
                as_tuple=False
            ).flatten()
            if query_indices.numel() > 0:
                summarized = self._summarize(concentration)
                alpha = prior_concentration + summarized
                q_probs = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                for query_index in query_indices.tolist():
                    concentration_slots[query_index] = alpha
                    q_slots[query_index] = q_probs
                if self.ablation == "instantaneous":
                    concentration = torch.zeros_like(concentration)

            previous_time = current_time

        posterior_concentration_grid = torch.stack(concentration_slots, dim=0)
        q_probs_grid = torch.stack(q_slots, dim=0)
        provider = _make_provider(
            t_grid,
            q_probs_grid,
            sample,
            self.tau_gumbel,
            self.hard_gumbel,
        )

        flat_slots = graph_ids * num_edges + physical_edge_ids
        physical_pair_evidence = event_evidence.new_zeros(
            batch_size * num_edges, self.edge_types
        ).index_add(0, flat_slots, event_evidence)
        physical_pair_evidence = physical_pair_evidence.reshape(
            batch_size, num_edges, self.edge_types
        )

        if event_evidence.numel() == 0:
            evidence_mean = event_evidence.new_zeros(())
            evidence_std = event_evidence.new_zeros(())
            evidence_max = event_evidence.new_zeros(())
            fraction_near_zero = event_evidence.new_ones(())
            evidence_by_relation = event_evidence.new_zeros(self.edge_types)
        else:
            evidence_mean = event_evidence.mean()
            evidence_std = event_evidence.std(unbiased=False)
            evidence_max = event_evidence.max()
            fraction_near_zero = (event_evidence < 1e-4).float().mean()
            evidence_by_relation = event_evidence.mean(dim=0)

        log_q = q_probs_grid.clamp_min(self.eps).log()
        entropy_by_relation = -q_probs_grid * log_q
        posterior_entropy = entropy_by_relation.sum(dim=-1)
        if q_probs_grid.size(0) > 1:
            temporal_variation = q_probs_grid.diff(dim=0).abs().mean()
        else:
            temporal_variation = q_probs_grid.new_zeros(())

        extras = {
            "event_evidence": event_evidence,
            "event_evidence_mean": evidence_mean,
            "event_evidence_std": evidence_std,
            "event_evidence_max": evidence_max,
            "event_evidence_by_relation": evidence_by_relation,
            "fraction_evidence_near_zero": fraction_near_zero,
            "physical_pair_evidence": physical_pair_evidence,
            "posterior_concentration_grid": posterior_concentration_grid,
            "posterior_total_concentration": posterior_concentration_grid.sum(-1),
            "q_probs_grid": q_probs_grid,
            "prior_probs": prior_probs,
            "prior_concentrations": prior_concentration,
            "posterior_entropy": posterior_entropy,
            "posterior_entropy_by_relation": entropy_by_relation,
            "age_kernel": self.get_age_kernel(),
            "decay_rates": self.decay_rates,
            "source_event_times": source_times,
            "physical_pair_indices": physical_pairs,
            "physical_edge_indices": physical_edge_ids,
            "event_edge_indices": valid,
            "lag_concentration_final": concentration,
            "q_temporal_variation": temporal_variation,
            "posterior_prior_l1": (
                q_probs_grid - prior_probs.unsqueeze(0)
            ).abs().mean(),
            # odeint_adjoint must receive dynamic closure tensors explicitly or
            # reconstruction gradients stop at the relation provider.
            "_adjoint_tensors": (q_probs_grid,) if q_probs_grid.requires_grad else (),
        }
        return provider, extras


def _make_provider(
    t_grid: torch.Tensor,
    q_probs_grid: torch.Tensor,
    sample: bool,
    tau_gumbel: float,
    hard_gumbel: bool,
):
    interpolator = TimeGridInterpolator(t_grid)

    def rel_type_provider(t_local: torch.Tensor):
        probs = interpolator.interp(q_probs_grid, t_local).float()
        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if sample:
            return gumbel_softmax_sample(
                probs.clamp_min(1e-8).log(),
                tau=tau_gumbel,
                hard=hard_gumbel,
            )
        return probs

    return rel_type_provider

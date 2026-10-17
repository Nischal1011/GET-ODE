# lib/at_transport_posterior.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.edge_latents import safe_softmax, gumbel_softmax_sample, TimeGridInterpolator

class ATTransportPosterior(nn.Module):
    """
    Builds q_phi(rel_type(t) | obs) as a time-dependent categorical distribution per structural edge.
    Uses encoder head-averaged attention on temporal edges as the event mass, then transports along lag bins
    so mass reaches lag=0 boundary over time.

    Output is in the NRI format:
      rel_type(t): [B, E, edge_types] where edge_types=2 (no-edge, edge)
    """

    def __init__(
        self,
        n_atoms: int,
        edge_types: int = 2,
        K_lag: int = 16,
        L: float = 1.0,
        v: float = 1.0,
        tau_gumbel: float = 1.0,
        hard_gumbel: bool = False,
        source_mode: str = "attention",
        eps: float = 1e-8,
    ):
        super().__init__()
        assert edge_types == 2, "This implementation assumes 2 edge types (no-edge / edge)."
        self.n_atoms = n_atoms
        self.edge_types = edge_types
        self.K_lag = K_lag
        self.L = L
        self.v = v
        self.tau_gumbel = tau_gumbel
        self.hard_gumbel = hard_gumbel
        if source_mode not in {"attention", "constant"}:
            raise ValueError(f"Unknown legacy AT source mode: {source_mode}")
        self.source_mode = source_mode
        self.eps = eps

        # Map boundary mass -> logits for two edge types.
        # Input is scalar mass per edge, output logits [no-edge, edge]
        self.boundary_to_logits = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, edge_types)
        )

        # Precompute mapping from (sender_atom, receiver_atom) -> edge_id in NRI off-diagonal ordering
        self.register_buffer("edge_sender", None, persistent=False)
        self.register_buffer("edge_receiver", None, persistent=False)
        self._build_edge_index_map()

    def _build_edge_index_map(self):
        n = self.n_atoms
        send = []
        recv = []
        for r in range(n):
            for s in range(n):
                if s == r:
                    continue
                recv.append(r)
                send.append(s)
        self.edge_receiver = torch.tensor(recv, dtype=torch.long)  # [E]
        self.edge_sender = torch.tensor(send, dtype=torch.long)    # [E]
        # Build hash map tensor via dictionary in python (fast enough once)
        self._pair_to_eid = {}
        for eid, (s, r) in enumerate(zip(send, recv)):
            self._pair_to_eid[(s, r)] = eid

    def _node_to_atom_ids(self, batch_en):
        """
        Reconstruct atom_id per node using batch_en.y (length per atom) and node ordering.
        Works per graph in batch. Returns:
          atom_id: [num_nodes] in [0..n_atoms-1]
        """
        if hasattr(batch_en, "object_id"):
            return batch_en.object_id.to(device=batch_en.x.device, dtype=torch.long)

        # Backward-compatible fallback for batches created before object_id existed.
        # batch_en.y shape: [num_graphs_in_batch * n_atoms]
        # each entry is number of nodes/observations for that atom in this graph
        y = batch_en.y  # LongTensor
        assert y.numel() % self.n_atoms == 0
        num_graphs = y.numel() // self.n_atoms

        atom_ids = torch.empty(batch_en.x.size(0), dtype=torch.long, device=batch_en.x.device)
        offset = 0
        idx = 0
        for g in range(num_graphs):
            for a in range(self.n_atoms):
                cnt = int(y[idx].item())
                atom_ids[offset:offset + cnt] = a
                offset += cnt
                idx += 1
        assert offset == batch_en.x.size(0)
        return atom_ids

    def _aggregate_temporal_attn_to_struct_edges(self, batch_en, edge_attn_temporal):
        """
        batch_en.edge_index: [2, E_temp] edges among observation-nodes (temporal graph)
        edge_attn_temporal: [E_temp, 1] head-averaged attention coefficients on those temporal edges

        Returns:
          mass_per_struct_edge: [B, E_struct] (sum of attention masses mapping to each structural edge)
          lag_per_temp_edge: [E_temp] in [0, L]
          batch_index_per_temp_edge: [E_temp] graph id in batch (0..B-1)
        """
        device = batch_en.x.device
        edge_index = batch_en.edge_index  # [2, E_temp]
        src = edge_index[0]
        dst = edge_index[1]

        atom_ids = self._node_to_atom_ids(batch_en)  # [num_nodes]
        src_atom = atom_ids[src]
        dst_atom = atom_ids[dst]

        # Determine which graph each node belongs to:
        # batch_en.batch is node->graph index in the mini-batch
        node_graph = batch_en.batch
        edge_graph = node_graph[dst]  # receiver's graph index

        B = int(node_graph.max().item()) + 1
        E_struct = self.edge_sender.numel()

        # Map (sender_atom, receiver_atom) -> eid
        # Here sender = src_atom, receiver = dst_atom
        # Build eid tensor
        eids = torch.empty(src_atom.numel(), dtype=torch.long, device=device)
        for k in range(src_atom.numel()):
            s = int(src_atom[k].item())
            r = int(dst_atom[k].item())
            if s == r:
                # self edges are not in NRI off-diagonal; just ignore (set to -1)
                eids[k] = -1
            else:
                eids[k] = self._pair_to_eid[(s, r)]

        valid = eids >= 0
        eids_valid = eids[valid]
        edge_graph_valid = edge_graph[valid]
        attn_valid = edge_attn_temporal[valid].squeeze(-1)  # [E_valid]

        # Aggregate masses: [B, E_struct]
        mass = torch.zeros(B, E_struct, device=device)
        mass.index_put_(
            (edge_graph_valid, eids_valid),
            attn_valid,
            accumulate=True
        )

        # Lag associated with temporal edge: use abs(edge_attr) (already delta time)
        lag = torch.clamp(torch.abs(batch_en.edge_attr).to(device), 0.0, self.L)  # [E_temp]
        return mass, lag, edge_graph

    def _bin_lags(self, lag_values: torch.Tensor):
        """
        lag_values: [...]
        returns: lag_bin in [0..K_lag-1]
        """
        # bins uniformly across [0, L]
        frac = torch.clamp(lag_values / max(self.L, 1e-12), 0.0, 1.0)
        bins = torch.floor(frac * (self.K_lag - 1 + 1e-6)).long()
        return torch.clamp(bins, 0, self.K_lag - 1)

    def _transport_down_to_zero(self, mu0, t_grid):
        """
        mu0: [B, E, K]
        t_grid: [T]
        returns mu_grid: [T, B, E, K]
        """
        device = mu0.device
        T = t_grid.numel()
        B, E, K = mu0.shape

        dl = self.L / max(self.K_lag - 1, 1)
        k_idx = torch.arange(K, device=device).view(1, 1, K)  # [1,1,K]

        # IMPORTANT: use list to avoid in-place writes on a tracked tensor
        mu_list = [mu0]  # mu_list[t] has shape [B,E,K]

        for ti in range(T - 1):
            dt = (t_grid[ti + 1] - t_grid[ti]).clamp(min=0.0)
            shift = (self.v * dt / max(dl, 1e-12))  # scalar tensor

            shift_floor = torch.floor(shift).long()
            shift_frac = (shift - shift_floor.float()).clamp(0.0, 1.0)

            mu = mu_list[-1]  # [B,E,K] (NOT a view into some big tensor)

            k_src = k_idx + shift_floor  # [1,1,K]
            k_src2 = k_src + 1  # [1,1,K]

            valid1 = ((k_src >= 0) & (k_src < K)).float()  # [1,1,K]
            valid2 = ((k_src2 >= 0) & (k_src2 < K)).float()  # [1,1,K]

            k_src_clamped = k_src.clamp(0, K - 1).expand(B, E, K)  # [B,E,K]
            k_src2_clamped = k_src2.clamp(0, K - 1).expand(B, E, K)  # [B,E,K]

            mu1 = mu.gather(-1, k_src_clamped) * valid1
            mu2 = mu.gather(-1, k_src2_clamped) * valid2

            mu_next = (1.0 - shift_frac) * mu1 + shift_frac * mu2
            mu_list.append(mu_next)

        mu_grid = torch.stack(mu_list, dim=0)  # [T,B,E,K]
        return mu_grid

    def build(self, batch_en, t_grid: torch.Tensor, sample: bool = True):
        """
        Returns:
          rel_type_provider(t): callable -> [B, E, 2] (soft or sampled)
          extras: dict with q_probs_grid, logits_grid, mu_grid, boundary_mass_grid
        """
        device = batch_en.x.device
        assert hasattr(batch_en, "edge_attn"), \
            "batch_en must carry head-averaged edge attention as batch_en.edge_attn from modified GTrans."

        # edge_attn over temporal edges: [E_temp, 1]
        edge_attn_temporal = batch_en.edge_attn  # computed by modified GTrans
        if self.source_mode == "constant":
            edge_attn_temporal = torch.ones_like(edge_attn_temporal)
        mass_struct, lag_temp, edge_graph = self._aggregate_temporal_attn_to_struct_edges(batch_en, edge_attn_temporal)

        # Create mu0 over lag bins by placing aggregated structural mass into bins.
        # Here we use lag bins based on temporal edge lag distribution by distributing mass across bins
        # using the average lag per structural edge computed via temporal edges (approx, efficient).

        B, E = mass_struct.shape
        K = self.K_lag
        mu0 = torch.zeros(B, E, K, device=device)

        # Approx lag-bin assignment: use mean temporal lag per graph (cheap baseline)
        # If you want per-edge lag, we can refine later using per-edge temporal lag aggregation.
        mean_lag = torch.mean(lag_temp).detach()  # scalar
        k0 = self._bin_lags(mean_lag)  # scalar bin
        mu0[:, :, k0] = mass_struct


        # Transport over time grid
        mu_grid = self._transport_down_to_zero(mu0, t_grid)  # [T,B,E,K]
        boundary_mass = mu_grid[..., 0]  # [T,B,E]

        # Map boundary mass -> logits for categorical edge types
        boundary_in = boundary_mass.unsqueeze(-1)  # [T,B,E,1]
        logits_grid = self.boundary_to_logits(boundary_in)  # [T,B,E,2]

        q_probs_grid = safe_softmax(logits_grid, dim=-1)  # [T,B,E,2]

        # sample or use soft probabilities
        if self.training and sample:
            rel_type_grid = gumbel_softmax_sample(logits_grid, tau=self.tau_gumbel, hard=self.hard_gumbel)
        else:
            rel_type_grid = q_probs_grid

        # Build fast provider via interpolation on logits
        interp = TimeGridInterpolator(t_grid)

        def rel_type_provider(t_local: torch.Tensor):
            # returns [B,E,2]
            logits_t = interp.interp(logits_grid, t_local)  # [B,E,2]
            probs_t = safe_softmax(logits_t, dim=-1)
            if self.training and sample:
                return gumbel_softmax_sample(logits_t, tau=self.tau_gumbel, hard=self.hard_gumbel)
            return probs_t

        extras = {
            "t_grid": t_grid,
            "mu_grid": mu_grid,
            "boundary_mass_grid": boundary_mass,
            "logits_grid": logits_grid,
            "q_probs_grid": q_probs_grid,
        }
        return rel_type_provider, extras


# Explicit name used in documentation while preserving old imports and checkpoints.
AttentionTransportPosterior = ATTransportPosterior

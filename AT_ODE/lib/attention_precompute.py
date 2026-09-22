import torch

def build_group_id_base(batch_g):
    """
    Receiver-group id per *structural* edge.
    Returns: group_id_base [E_base], N (nodes/graph), G (graphs/batch)
    """
    ei   = batch_g.edge_index            # [2, E_base]
    dst  = ei[1]                         # [E_base]
    ptr  = batch_g.ptr                   # [G+1]
    nb   = batch_g.batch                 # [N_base] graph id per node
    assert ptr is not None and nb is not None

    ge   = nb[dst]                       # [E_base] graph id of dst
    N    = int((ptr[1] - ptr[0]).item()) # nodes per graph
    iloc = dst - ptr[ge]                 # [E_base] local dst in that graph
    group_id_base = ge * N + iloc        # [E_base] in [0, G*N-1]
    return group_id_base, N, (ptr.numel()-1)

def bus_of_event_batch(batch_en):
    """
    Map encoder *event node* -> bus id (local in its graph).
    Returns: [N_ev_total] long.
    """
    ptr_ev = batch_en.ptr     # [G+1]
    y_cat  = batch_en.y       # [G*N]
    assert ptr_ev is not None and y_cat is not None
    G = ptr_ev.numel() - 1
    N = y_cat.numel() // G
    out = []
    off = 0
    for _ in range(G):
        y_g = y_cat[off:off+N]
        out.append(torch.repeat_interleave(torch.arange(N, device=y_cat.device), y_g))
        off += N
    return torch.cat(out, dim=0)


@torch.no_grad()
def build_attn_event_eL_base(
    batch_en,                  # encoder-expanded PyG Data (has .pos [N_ev], .y [num_buses], etc.)
    edge_index_struct,         # [2, E_base] structural graph edges (base, no time expansion)
    encoder_model,             # the encoder (to pull/compute edge-attn)
    L: int,                    # number of lag bins
    dtype=torch.float32,
):
    """
    Returns:
      attn_event_eL_base: Tensor [E_base, L] on the same device as edge_index_struct
    """
    dev = edge_index_struct.device

    # Times for encoder event nodes and per-bus counts
    # (batch_en carries these; do NOT read from structural graph)
    t  = batch_en.pos.to(device=dev)           # [N_event_nodes]
    y_counts = batch_en.y.to(device=dev)       # [num_buses]

    # 1) Get encoder event-edge attentions (alpha over time-expanded edges).
    #    If you already captured attention in the encoder, fetch it; else compute it here.
    #    For example (if you captured last-layer edge attention):
    alpha_ev, edge_index_ev = encoder_model.get_last_event_edge_attention()  # both on CPU typically
    if alpha_ev is None or edge_index_ev is None:
        # If you need to run a forward or otherwise compute alpha_ev, do it here.
        # But most of your pipeline sets capture=True and has these ready.
        raise RuntimeError("Encoder edge attention not captured; enable capture before calling precompute.")

    # Move to the right device / dtype
    alpha_ev       = alpha_ev.to(device=dev, dtype=dtype)           # [E_enc]
    edge_index_ev  = edge_index_ev.to(device=dev)                   # [2, E_enc]

    # 2) Map event edges to (sender bus, receiver bus) and compute lag bin indices using batch_en.pos (t)
    #    Build a per-structural-edge histogram over L bins by accumulating alpha_ev of its event-edges.
    #    Pseudocode outline:
    #
    #    - For each event edge e' = (i -> j) in edge_index_ev:
    #         * sender bus id = bus_of_node[i], receiver bus id = bus_of_node[j]
    #           (you can infer bus ids by expanding y_counts into a per-node bus index)
    #         * lag = t[j] - t[i], then bin to b in [0, L-1]
    #         * find the structural edge (bus_of_node[i] -> bus_of_node[j]) in edge_index_struct
    #         * add alpha_ev[e'] into histogram[that_structural_edge, b]
    #
    #    - Finally, row-normalize histograms over L.

    # Build node->bus mapping from y_counts (repeat bus id per its node count)
    bus_ids = torch.arange(y_counts.numel(), device=dev).repeat_interleave(y_counts)  # [N_event_nodes]
    assert bus_ids.numel() == t.numel(), "Mismatch between repeated bus ids and encoder nodes."

    send_ev = edge_index_ev[0]  # [E_enc]
    recv_ev = edge_index_ev[1]  # [E_enc]
    bus_s = bus_ids[send_ev]    # [E_enc]
    bus_r = bus_ids[recv_ev]    # [E_enc]

    # Compute lag (you already normalized times elsewhere; no-op if already in your desired units)
    lag = t[recv_ev] - t[send_ev]  # [E_enc]

    # Bin lags into L bins. Example: uniform over observed range
    lag_min = lag.min()
    lag_max = lag.max()
    bins = torch.linspace(lag_min, lag_max + 1e-12, L + 1, device=dev)
    bin_idx = torch.bucketize(lag, bins) - 1    # [E_enc], in [0, L-1]
    bin_idx.clamp_(0, L - 1)

    # Build a map from (bus_s, bus_r) -> structural edge id
    E_base = int(edge_index_struct.size(1))
    key_sr = edge_index_struct[0] * y_counts.numel() + edge_index_struct[1]
    key_ev = bus_s * y_counts.numel() + bus_r
    # Make a hashtable by scatter (O(E_base + E_enc))
    # For lookup: we need, for each event edge, its structural edge index.
    idx_struct = torch.full((y_counts.numel() * y_counts.numel(),), -1, device=dev, dtype=torch.long)
    idx_struct[key_sr] = torch.arange(E_base, device=dev)

    e_ids = idx_struct[key_ev]   # [E_enc], -1 means event edge doesn’t correspond to any structural edge (shouldn’t happen with your build)

    # Accumulate alpha_ev into hist[E_base, L]
    hist = torch.zeros(E_base, L, device=dev, dtype=dtype)
    mask = (e_ids >= 0)
    hist.index_put_((e_ids[mask], bin_idx[mask]), alpha_ev[mask], accumulate=True)

    # Normalize per edge over bins
    hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-9)

    return hist  # [E_base, L]


def init_P_prev_uniform(edge_index_struct: torch.Tensor, n_nodes: int, device=None):
    """
    Uniform over incoming edges per receiver (group).
    Returns: Tensor [E_base] with group-normalized masses.
    """
    if device is None:
        device = edge_index_struct.device
    E = edge_index_struct.size(1)
    dst = edge_index_struct[1]
    ones = torch.ones(E, device=device)
    deg  = torch.zeros(n_nodes, device=device).index_add(0, dst, ones)  # in-degree per receiver
    return 1.0 / deg[dst].clamp_min(1e-9)  # [E]
'''
Shared per-node, non-graph continuous-time machinery used by both the ODE-RNN and Latent-ODE
baselines (lib/baseline_odernn.py, lib/baseline_latent_ode.py). Reuses lib/diffeq_solver.py's
ODEFunc (a plain, non-graph ODE function wrapper already present in the codebase but unused by
LG-ODE itself) and lib/utils.py's create_net.

Both baselines plug into the EXISTING encoder/decoder Data objects produced by
CorrectedParseData / IEEE39ParseData -- they simply ignore edge_index/edge_attr/edge_same
(the graph fields), using only x (observed features), pos (observed times), and y (per-object
observation counts), which is exactly "provide observation times, observed values, and masks;
do not provide graph adjacency or neighboring-node states."

Every dataset in this project discretizes time onto a small shared grid (springs/charged: 60
steps; IEEE39: 60 regular steps), so an observation time and a decoder query time for the same
underlying instant are numerically identical floats (both derived from the same source array
via the same normalization) -- this lets the union of observed + query times per batch be built
losslessly with a plain torch.unique + searchsorted, no tolerance/rounding needed (verified
empirically before relying on it, see the smoke tests this module's callers run).
'''
import torch
import torch.nn as nn
from torchdiffeq import odeint

import lib.utils as utils
from lib.diffeq_solver import ODEFunc


def make_ode_func(hidden_dim, n_units=100):
    net = utils.create_net(hidden_dim, hidden_dim, n_layers=1, n_units=n_units, nonlinear=nn.Tanh)
    return ODEFunc(input_dim=hidden_dim, latent_dim=hidden_dim, ode_func_net=net)


def run_batched_ode_rnn(ode_func, gru_cell, input_proj, x, pos, y, decoder_time_steps,
                         hidden_dim, device, method='rk4', h0=None):
    '''
    x: [N_obs, D] observed features, flattened across all M independent (trajectory, node)
       sequences in this batch.
    pos: [N_obs] observed times, row-aligned with x.
    y: [M] number of observed rows belonging to each of the M sequences (cumulative sum over y
       gives each sequence's row range in x/pos) -- exactly the same "y" field CorrectedParseData/
       IEEE39ParseData already build for the graph encoder, just reused without the graph.
    decoder_time_steps: [T_Q] query times shared by every sequence in this batch.
    h0: optional [M, hidden_dim] initial hidden state (zeros if not given).

    Returns: h_at_query [M, T_Q, hidden_dim] -- hidden state at each query time, per sequence,
    in the same order as decoder_time_steps.
    '''
    M = y.shape[0]
    all_times, _ = torch.cat([pos, decoder_time_steps]).unique(sorted=True), None
    T_master = all_times.shape[0]

    seq_id = torch.repeat_interleave(torch.arange(M, device=device), y)
    time_idx = torch.searchsorted(all_times, pos)

    obs_dense = torch.zeros(M, T_master, x.shape[-1], device=device)
    obs_mask = torch.zeros(M, T_master, dtype=torch.bool, device=device)
    obs_dense[seq_id, time_idx] = x
    obs_mask[seq_id, time_idx] = True

    query_idx = torch.searchsorted(all_times, decoder_time_steps)

    h = torch.zeros(M, hidden_dim, device=device) if h0 is None else h0
    outputs = torch.zeros(M, T_master, hidden_dim, device=device)

    prev_t = all_times[0]
    for i in range(T_master):
        cur_t = all_times[i]
        if i > 0 and (cur_t - prev_t).abs() > 1e-8:
            t_span = torch.stack([prev_t, cur_t])
            h = odeint(ode_func, h, t_span, method=method)[-1]
        mask_i = obs_mask[:, i]
        if mask_i.any():
            x_in = input_proj(obs_dense[:, i])
            h_updated = gru_cell(x_in, h)
            h = torch.where(mask_i.unsqueeze(-1), h_updated, h)
        outputs[:, i] = h
        prev_t = cur_t

    return outputs[:, query_idx]


def run_batched_ode_rnn_backward(ode_func, gru_cell, input_proj, x, pos, y, t0,
                                  hidden_dim, device, method='rk4'):
    '''
    Canonical Latent-ODE encoder direction (Rubanova et al. 2019): processes observations in
    REVERSE chronological order, integrating the hidden state backward in time between
    consecutive observations (torchdiffeq's odeint natively integrates correctly when given a
    decreasing time span, no gradient negation needed) and applying the GRU update at each
    observation in its correct temporal position, ending with one final backward step to t0 --
    rather than a forward pass followed by a single big backward jump at the end (the earlier
    simplification this replaces, see lib/baseline_latent_ode.py). Returns h at t0, [M, hidden_dim].
    '''
    M = y.shape[0]
    obs_times = pos.unique(sorted=True)  # ascending
    T_obs = obs_times.shape[0]

    seq_id = torch.repeat_interleave(torch.arange(M, device=device), y)
    time_idx = torch.searchsorted(obs_times, pos)

    obs_dense = torch.zeros(M, T_obs, x.shape[-1], device=device)
    obs_mask = torch.zeros(M, T_obs, dtype=torch.bool, device=device)
    obs_dense[seq_id, time_idx] = x
    obs_mask[seq_id, time_idx] = True

    h = torch.zeros(M, hidden_dim, device=device)
    prev_t = obs_times[-1]

    for i in range(T_obs - 1, -1, -1):
        cur_t = obs_times[i]
        if (prev_t - cur_t).abs() > 1e-8:
            t_span = torch.stack([prev_t, cur_t])
            h = odeint(ode_func, h, t_span, method=method)[-1]
        mask_i = obs_mask[:, i]
        if mask_i.any():
            x_in = input_proj(obs_dense[:, i])
            h_updated = gru_cell(x_in, h)
            h = torch.where(mask_i.unsqueeze(-1), h_updated, h)
        prev_t = cur_t

    if (prev_t - t0).abs() > 1e-8:
        t_span = torch.stack([prev_t, t0])
        h = odeint(ode_func, h, t_span, method=method)[-1]

    return h

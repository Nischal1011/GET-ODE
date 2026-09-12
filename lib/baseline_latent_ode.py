'''
Latent-ODE baseline (Rubanova et al. 2019): per-node, non-graph VAE. Each node's observations
are encoded independently into an approximate posterior q_phi(z0|O) (mean/std of the latent
initial state), z0 is sampled and evolved by a neural ODE, and a linear decoder maps the
resulting latent trajectory back to observation space. No physical edges or neighboring
observations are used -- same encoder, ODE function, and decoder parameters are shared across
every node (particle/generator), exactly like ODE-RNN (lib/baseline_odernn.py), but the
node-independent processing here compresses context into an inferred initial condition and
generates the trajectory from it, rather than updating state directly at every observation.

Reuses VAE_Baseline's compute_all_losses UNCHANGED (same pattern as LatentGraphODE / LG-ODE
itself) -- only get_reconstruction is implemented, producing the same
info["first_point"] = (mean, std, z0_sample) structure the inherited KL computation expects.

Encoder design note: the "textbook" Latent-ODE encoder runs its ODE-RNN BACKWARDS in time (from
the last observation to the first) so z0 directly represents the state at t=0. This
implementation instead runs the encoder FORWARD (same direction as ODE-RNN's own encoding) and
then does one extra backward ODE solve from the last-observed time to t=0 (or to the
context/forecast boundary time for extrapolation, matching every other model in this project's
z0-reference convention) to align the summary with a t=0 initial condition. This is a
simplification for implementation tractability, not the original paper's exact encoder -- flagged
here rather than silently deviating.
'''
import torch
import torch.nn as nn
from torchdiffeq import odeint

import lib.utils as utils
from lib.base_models import VAE_Baseline
from lib.nongraph_ode import make_ode_func, run_batched_ode_rnn


class LatentODEBaseline(VAE_Baseline):

    def __init__(self, input_dim, latent_dim, hidden_dim, z0_prior, obsrv_std, device):
        super(LatentODEBaseline, self).__init__(
            input_dim=input_dim, latent_dim=latent_dim, z0_prior=z0_prior, device=device, obsrv_std=obsrv_std)

        self.hidden_dim = hidden_dim

        self.enc_input_proj = nn.Linear(input_dim, hidden_dim)
        self.enc_gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.enc_ode_func = make_ode_func(hidden_dim)
        self.z0_head = nn.Linear(hidden_dim, latent_dim * 2)

        self.gen_ode_func = make_ode_func(latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

        utils.init_network_weights(self.enc_input_proj)
        utils.init_network_weights(self.z0_head)
        utils.init_network_weights(self.decoder)

    def encode_z0(self, batch_en):
        x, pos, y = batch_en.x, batch_en.pos, batch_en.y
        query_times = pos.unique(sorted=True)

        h_at_obs = run_batched_ode_rnn(
            self.enc_ode_func, self.enc_gru_cell, self.enc_input_proj, x, pos, y, query_times,
            self.hidden_dim, self.device)
        h_end = h_at_obs[:, -1]  # state at query_times[-1], the last observation in this batch

        t_last = query_times[-1]
        if t_last.item() > 1e-8:
            t_span = torch.stack([t_last, torch.zeros_like(t_last)])
            h0 = odeint(self.enc_ode_func, h_end, t_span, method='rk4')[-1]
        else:
            h0 = h_end

        z0_params = self.z0_head(h0)
        mean = z0_params[..., :self.latent_dim]
        std = z0_params[..., self.latent_dim:].abs()
        return mean, std

    def get_reconstruction(self, batch_en, batch_de, batch_g, n_traj_samples=1, run_backwards=True):
        mean_z0, std_z0 = self.encode_z0(batch_en)

        means_z0 = mean_z0.repeat(n_traj_samples, 1, 1)
        sigmas_z0 = std_z0.repeat(n_traj_samples, 1, 1)
        z0_sample = utils.sample_standard_gaussian(means_z0, sigmas_z0)  # [n_traj_samples, M, latent_dim]

        time_steps_to_predict = batch_de["time_steps"]
        ispadding = time_steps_to_predict[0] != 0
        if ispadding:
            solve_times = torch.cat([torch.zeros(1, device=time_steps_to_predict.device), time_steps_to_predict])
        else:
            solve_times = time_steps_to_predict

        n_traj = z0_sample.size(1)
        z0_flat = z0_sample.reshape(-1, self.latent_dim)
        sol = odeint(self.gen_ode_func, z0_flat, solve_times, method='rk4')  # [T, n_traj_samples*n_traj, latent_dim]
        if ispadding:
            sol = sol[1:]

        sol = sol.permute(1, 0, 2).reshape(n_traj_samples, n_traj, len(time_steps_to_predict), self.latent_dim)
        pred_x = self.decoder(sol)  # [n_traj_samples, n_traj, T_Q, D]

        all_extra_info = {
            "first_point": (torch.unsqueeze(mean_z0, 0), torch.unsqueeze(std_z0, 0), z0_sample),
            "latent_traj": sol.detach()
        }
        return pred_x, all_extra_info, None

'''
ODE-RNN baseline (Rubanova et al. 2019), per lib/nongraph_ode.py's shared batched runner.

One shared ODE-RNN cell processes every node (particle/generator) as an independent irregular
sequence -- no graph adjacency, no neighboring-node states. Between observations the hidden
state evolves continuously via a neural ODE; at each observation it's updated with a GRUCell.
Unlike Latent-ODE (lib/baseline_latent_ode.py), there is no separate "infer z0 then generate"
step -- the recurrently-updated hidden state itself is decoded directly at every query time,
so it's a deterministic reconstruction model (no z0 posterior, hence no KL term).

Reuses VAE_Baseline's get_gaussian_likelihood/get_mse (they don't depend on the KL/z0_prior
machinery) purely to keep the loss/metric computation identical to every other model in this
project; compute_all_losses is otherwise a fresh, non-variational implementation.
'''
import torch
import torch.nn as nn

import lib.utils as utils
from lib.base_models import VAE_Baseline
from lib.nongraph_ode import make_ode_func, run_batched_ode_rnn


class ODERNNBaseline(VAE_Baseline):

    def __init__(self, input_dim, hidden_dim, obsrv_std, device):
        super(ODERNNBaseline, self).__init__(
            input_dim=input_dim, latent_dim=hidden_dim, z0_prior=None, device=device, obsrv_std=obsrv_std)

        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.ode_func = make_ode_func(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        utils.init_network_weights(self.input_proj)
        utils.init_network_weights(self.decoder)

    def get_reconstruction(self, batch_en, batch_de):
        x, pos, y = batch_en.x, batch_en.pos, batch_en.y
        decoder_time_steps = batch_de["time_steps"]

        h_at_query = run_batched_ode_rnn(
            self.ode_func, self.gru_cell, self.input_proj, x, pos, y, decoder_time_steps,
            self.hidden_dim, self.device)

        pred_x = self.decoder(h_at_query)  # [M, T_Q, D]
        return pred_x.unsqueeze(0)  # [1, M, T_Q, D] -- n_traj_samples=1, deterministic

    def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                            n_traj_samples=1, kl_coef=1.):
        pred_x = self.get_reconstruction(batch_dict_encoder, batch_dict_decoder)

        truth = batch_dict_decoder["data"]
        mask = batch_dict_decoder["mask"]

        rec_likelihood = self.get_gaussian_likelihood(truth, pred_x, None, mask=mask)
        mse_val = self.get_mse(truth, pred_x, mask=mask)

        loss = -torch.mean(rec_likelihood)

        return {
            "loss": loss,
            "likelihood": torch.mean(rec_likelihood).data.item(),
            "mse": torch.mean(mse_val).data.item(),
            "kl_first_p": 0.0,
            "std_first_p": 0.0,
        }

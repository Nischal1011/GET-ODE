'''
Wraps lib/gil_ode.py's GILODEModel with the same likelihood/MSE computation every other model
in this project uses (reusing VAE_Baseline's get_gaussian_likelihood/get_mse, which don't
depend on the KL/z0_prior machinery -- same reuse pattern as lib/baseline_odernn.py). GIL-ODE is
deterministic (no z0 posterior), so there is no KL term.
'''
import torch

from lib.base_models import VAE_Baseline
from lib.gil_ode import GILODEModel
from lib.gil_dataset import prepare_gil_batch


class GILODEBaseline(VAE_Baseline):

    def __init__(self, input_dim, hidden_dim, num_atoms, dataset, obsrv_std, device):
        super(GILODEBaseline, self).__init__(
            input_dim=input_dim, latent_dim=hidden_dim, z0_prior=None, device=device, obsrv_std=obsrv_std)
        self.num_atoms = num_atoms
        self.dataset = dataset
        self.core = GILODEModel(input_dim, hidden_dim, num_atoms, device)

    def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                            n_traj_samples=1, kl_coef=1.):
        dense, mask, grid_times, S, c = prepare_gil_batch(
            batch_dict_encoder, batch_dict_decoder, batch_dict_graph, self.num_atoms, self.dataset, self.device)

        decoder_time_steps = batch_dict_decoder["time_steps"]
        pred = self.core(dense, mask, grid_times, decoder_time_steps, S, c)  # [B, N, T_Q, D]

        B, N, T_Q, D = pred.shape
        pred_flat = pred.reshape(B * N, T_Q, D).unsqueeze(0)  # [1, M, T_Q, D]

        truth = batch_dict_decoder["data"]
        target_mask = batch_dict_decoder["mask"]

        rec_likelihood = self.get_gaussian_likelihood(truth, pred_flat, None, mask=target_mask)
        mse_val = self.get_mse(truth, pred_flat, mask=target_mask)

        loss = -torch.mean(rec_likelihood)

        return {
            "loss": loss,
            "likelihood": torch.mean(rec_likelihood).data.item(),
            "mse": torch.mean(mse_val).data.item(),
            "kl_first_p": self.core.ode_func.alpha.data.item(),  # repurposed slot: track alpha's growth
            "std_first_p": 0.0,
        }

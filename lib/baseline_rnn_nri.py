'''
Wraps lib/nri_baseline.py's NRIBaseline with the same likelihood/MSE computation every other
model in this project uses (reusing VAE_Baseline's get_gaussian_likelihood/get_mse, which don't
depend on the KL/z0_prior machinery -- same reuse pattern as lib/baseline_odernn.py). The only
addition is the relation-type KL term (posterior over K edge_types vs. a uniform prior),
active only in extrapolation mode where the relation encoder actually runs.
'''
import torch

from lib.base_models import VAE_Baseline
from lib.nri_baseline import NRIBaseline


class RNNNRIBaseline(VAE_Baseline):

    def __init__(self, input_dim, num_atoms, hidden_dim, edge_types, mode, obsrv_std, device):
        super(RNNNRIBaseline, self).__init__(
            input_dim=input_dim, latent_dim=hidden_dim, z0_prior=None, device=device, obsrv_std=obsrv_std)
        self.core = NRIBaseline(input_dim, num_atoms, hidden_dim, edge_types, mode, obsrv_std, device)

    def compute_all_losses(self, batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                            n_traj_samples=1, kl_coef=1.):
        pred_x, rel_logits = self.core(batch_dict_encoder, batch_dict_decoder)

        truth = batch_dict_decoder["data"]
        mask = batch_dict_decoder["mask"]

        rec_likelihood = self.get_gaussian_likelihood(truth, pred_x, None, mask=mask)
        mse_val = self.get_mse(truth, pred_x, mask=mask)
        rel_kl = self.core.relation_kl(rel_logits)

        loss = -torch.mean(rec_likelihood) + kl_coef * rel_kl

        return {
            "loss": loss,
            "likelihood": torch.mean(rec_likelihood).data.item(),
            "mse": torch.mean(mse_val).data.item(),
            "kl_first_p": rel_kl.data.item(),
            "std_first_p": 0.0,
        }

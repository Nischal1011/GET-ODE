from lib.likelihood_eval import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
import torch

from lib.edge_latents import kl_categorical


class VAE_Baseline(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        z0_prior,
        device,
        obsrv_std=0.01,
    ):
        super(VAE_Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
        self.z0_prior = z0_prior

        # Optional (set later on model instance)
        # self.edge_prior_mode = "uniform" or "graph"
        # self.edge_prior_p = 0.9
        # self.kl_edge_coef = 1.0

    def get_gaussian_likelihood(self, truth, pred_y, temporal_weights, mask):
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(
            pred_y,
            truth_repeated,
            obsrv_std=self.obsrv_std,
            mask=mask,
            temporal_weights=temporal_weights,
        )
        log_density_data = log_density_data.permute(1, 0)
        log_density = torch.mean(log_density_data, 1)
        return log_density

    def get_mse(self, truth, pred_y, mask=None):
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = compute_mse(pred_y, truth_repeated, mask=mask)
        return torch.mean(log_density_data)

    def compute_all_losses(
        self,
        batch_dict_encoder,
        batch_dict_decoder,
        batch_dict_graph,
        n_traj_samples=1,
        kl_coef=1.0,
    ):
        # 1) Reconstruction
        pred_y, info, temporal_weights = self.get_reconstruction(
            batch_dict_encoder,
            batch_dict_decoder,
            batch_dict_graph,
            n_traj_samples=n_traj_samples,
        )

        # 2) KL for z0
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_distr = Normal(fp_mu, fp_std)

        assert torch.sum(fp_std < 0) == 0.0

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)  # shape [S, B, latent] typically

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is NaN!")

        # Mean over latent dims and objects (keep sample dim)
        kldiv_z0 = torch.mean(kldiv_z0, (1, 2))

        # 3) KL for edges (Option 2)
        kldiv_edge = torch.tensor(0.0, device=self.device)
        evidence_l1 = torch.tensor(0.0, device=self.device)
        edge_extras = info.get("edge_extras", None)

        if edge_extras is not None:
            # Concentration normalization and categorical KL stay in float32
            # under AMP to avoid small-evidence underflow.
            q_probs_grid = edge_extras["q_probs_grid"].float()  # [T, B, E, R]

            edge_prior_mode = getattr(self, "edge_prior_mode", "uniform")
            edge_prior_p = float(getattr(self, "edge_prior_p", 0.9))

            if "prior_probs" in edge_extras:
                p_probs = edge_extras["prior_probs"].float()
                if p_probs.ndim == 3:
                    p_probs = p_probs.unsqueeze(0).expand_as(q_probs_grid)
            elif edge_prior_mode == "uniform":
                p_probs = torch.ones_like(q_probs_grid) * (1.0 / q_probs_grid.size(-1))
            else:
                # Graph-informed prior: batch_dict_graph should be [B, E] with 0/1
                g = batch_dict_graph
                if isinstance(g, dict):
                    g = g.get("graph", g)
                g = g.long()

                # Build p(edge)=p for observed, else (1-p)
                p_edge = torch.where(
                    g == 1,
                    torch.full_like(g, edge_prior_p, dtype=torch.float),
                    torch.full_like(g, 1.0 - edge_prior_p, dtype=torch.float),
                )
                p_no = 1.0 - p_edge

                p_probs = torch.stack([p_no, p_edge], dim=-1)  # [B, E, 2]
                p_probs = p_probs.unsqueeze(0).repeat(q_probs_grid.size(0), 1, 1, 1)  # [T,B,E,2]

            kl_tbe = kl_categorical(q_probs_grid, p_probs)  # [T, B, E]
            kldiv_edge = kl_tbe.mean()

            if "event_evidence" in edge_extras:
                event_evidence = edge_extras["event_evidence"].float()
                if event_evidence.numel() > 0:
                    evidence_l1 = event_evidence.mean()

        # 4) Likelihood + MSE
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict_decoder["raw"],
            pred_y,
            temporal_weights,
            mask=batch_dict_decoder["mask"],
        )

        mse = self.get_mse(
            batch_dict_decoder["raw"],
            pred_y,
            mask=batch_dict_decoder["mask"],
        )

        # 5) Loss (ELBO-style)
        kl_edge_coef = float(getattr(self, "kl_edge_coef", 1.0))
        evidence_l1_coef = float(getattr(self, "evidence_l1_coef", 0.0))

        # rec_likelihood: [n_traj_samples], kldiv_z0: [n_traj_samples]
        loss = -torch.logsumexp(
            rec_likelihood
            - kl_coef * kldiv_z0
            - kl_edge_coef * kldiv_edge
            - evidence_l1_coef * evidence_l1,
            0,
        )

        if torch.isnan(loss):
            loss = -torch.mean(
                rec_likelihood
                - kl_coef * kldiv_z0
                - kl_edge_coef * kldiv_edge
                - evidence_l1_coef * evidence_l1,
                0,
            )

        # 6) Results
        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).item()
        results["mse"] = torch.mean(mse).item()
        results["kl_first_p"] = torch.mean(kldiv_z0).detach().item()
        results["std_first_p"] = torch.mean(fp_std).detach().item()
        results["pred_samples"] = pred_y.detach()
        results["kl_edge"] = float(kldiv_edge.detach().item())
        results["evidence_l1"] = float(evidence_l1.detach().item())

        if edge_extras is not None:
            scalar_diagnostics = {
                "event_evidence_mean": "evidence_mean",
                "event_evidence_std": "evidence_std",
                "event_evidence_max": "evidence_max",
                "fraction_evidence_near_zero": "fraction_evidence_near_zero",
                "q_temporal_variation": "q_temporal_variation",
                "posterior_prior_l1": "posterior_prior_l1",
                "aggregate_nri_message_norm": "nri_message_norm",
                "latent_derivative_norm": "latent_derivative_norm",
            }
            for source_key, result_key in scalar_diagnostics.items():
                value = edge_extras.get(source_key)
                if value is not None:
                    results[result_key] = float(value.detach().float().mean().item())

        return results

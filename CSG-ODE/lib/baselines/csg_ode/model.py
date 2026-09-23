"""End-to-end CSG-ODE latent-variable model."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor, nn

from .config import CSGODEConfig
from .csode_func import ControlSynthGraphODEFunc
from .encoder import CSGEncoder
from .metrics import gaussian_log_likelihood, metric_bundle
from .solver import solve_joint_ode


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class CSGODE(nn.Module):
    """CSG-ODE with a Gaussian z0 posterior and coupled z/c decoder dynamics."""

    def __init__(self, config: CSGODEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CSGEncoder(config)
        self.ode_function = ControlSynthGraphODEFunc(config)
        self.decoder = nn.Linear(config.latent_dim, config.input_dim)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.decoder.bias)

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        n_traj_samples: int | None = None,
        sample_posterior: bool = True,
        return_diagnostics: bool = False,
        profile: bool = False,
    ) -> dict[str, Any]:
        samples = n_traj_samples or self.config.n_traj_samples
        device = batch["observed_values"].device
        timings: dict[str, float] = {}

        if profile:
            _synchronize(device)
        start = time.perf_counter()
        encoded = self.encoder(batch, return_diagnostics=return_diagnostics)
        if profile:
            _synchronize(device)
            timings["encoder_seconds"] = time.perf_counter() - start

        mean = encoded["mean"]
        std = encoded["std"]
        if sample_posterior:
            noise = torch.randn(samples, *mean.shape, dtype=mean.dtype, device=mean.device)
            z0 = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        else:
            samples = 1
            z0 = mean.unsqueeze(0)

        if self.config.augment_dim:
            augmentation = z0.new_zeros(*z0.shape[:-1], self.config.augment_dim)
            z0_augmented = torch.cat([z0, augmentation], dim=-1)
        else:
            z0_augmented = z0

        batch_size = mean.shape[0]
        flat_z0 = z0_augmented.reshape(
            samples * batch_size, self.config.num_nodes, self.config.dynamics_dim
        )
        adjacency = batch["original_adjacency"].repeat(samples, 1, 1)
        edge_type = batch["edge_type"].repeat(samples, 1, 1)
        control0 = self.ode_function.initial_control(flat_z0, adjacency, edge_type)
        y0 = torch.cat([flat_z0, control0], dim=-1)
        target_times = batch["target_times"].repeat(samples, 1)
        target_valid = batch["target_time_valid_mask"].repeat(samples, 1)

        if profile:
            _synchronize(device)
        start = time.perf_counter()
        trajectory = solve_joint_ode(
            self.ode_function,
            y0,
            target_times,
            target_valid,
            adjacency,
            edge_type,
            method=self.config.solver,
            rtol=self.config.rtol,
            atol=self.config.atol,
        )
        if profile:
            _synchronize(device)
            timings["ode_seconds"] = time.perf_counter() - start

        latent_augmented = trajectory[..., : self.config.dynamics_dim]
        latent = latent_augmented[..., : self.config.latent_dim]
        latent = latent.reshape(
            samples,
            batch_size,
            latent.shape[1],
            self.config.num_nodes,
            self.config.latent_dim,
        )
        if profile:
            _synchronize(device)
        start = time.perf_counter()
        predictions = self.decoder(latent)
        if profile:
            _synchronize(device)
            timings["decoder_seconds"] = time.perf_counter() - start

        result: dict[str, Any] = {
            "predictions": predictions,
            "posterior_mean": mean,
            "posterior_std": std,
            "kl": encoded["kl"],
            "encoded": encoded,
            "latent_trajectory": latent,
            "timings": timings,
        }
        if return_diagnostics:
            terms = self.ode_function.derivative_terms(y0, adjacency, edge_type)
            result["ode_diagnostics"] = {
                "linear_norm": terms["linear"].norm().detach(),
                "nonlinear_norms": torch.stack(
                    [term.norm().detach() for term in terms["nonlinear_terms"]]
                ),
                "control_norm": terms["control"].norm().detach(),
                "control_dot_norm": terms["control_dot"].norm().detach(),
                "z_dot_norm": terms["z_dot"].norm().detach(),
                "joint_trajectory": trajectory.detach(),
                "decoded_trajectory": predictions.detach(),
            }
        return result

    def compute_all_losses(
        self,
        batch: dict[str, Tensor],
        *,
        kl_coef: float = 1.0,
        n_traj_samples: int | None = None,
        sample_posterior: bool = True,
        return_diagnostics: bool = False,
        profile: bool = False,
    ) -> dict[str, Any]:
        output = self.forward(
            batch,
            n_traj_samples=n_traj_samples,
            sample_posterior=sample_posterior,
            return_diagnostics=return_diagnostics,
            profile=profile,
        )
        predictions = output["predictions"]
        targets = batch["target_values"]
        target_mask = batch["target_mask"]
        likelihood_by_sample = gaussian_log_likelihood(
            predictions, targets, target_mask, self.config.obsrv_std
        )
        kl = output["kl"]
        objective = likelihood_by_sample - float(kl_coef) * kl
        loss = -torch.logsumexp(objective, dim=0)
        if not torch.isfinite(loss):
            loss = -objective.mean()
        metrics = metric_bundle(
            predictions,
            targets,
            target_mask,
            batch["target_unobserved_mask"],
        )
        return {
            **output,
            **metrics,
            "loss": loss,
            "likelihood": likelihood_by_sample.mean(),
            "reconstruction_nll": -likelihood_by_sample.mean(),
            "kl_first_point": kl,
            "kl_coef": torch.as_tensor(kl_coef, device=loss.device),
            "posterior_std_mean": output["posterior_std"].mean(),
        }

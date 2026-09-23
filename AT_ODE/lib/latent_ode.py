from lib.base_models import VAE_Baseline
import lib.utils as utils
import torch


class LatentGraphODE(VAE_Baseline):
    def __init__(
        self,
        input_dim,
        latent_dim,
        encoder_z0,
        decoder,
        diffeq_solver,
        z0_prior,
        device,
        obsrv_std=None,
    ):
        super(LatentGraphODE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            z0_prior=z0_prior,
            device=device,
            obsrv_std=obsrv_std,
        )

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.latent_dim = latent_dim

        # These will be attached later (Option 2)
        # self.edge_posterior = ...
        # self.edge_prior_mode = ...
        # self.edge_prior_p = ...
        # self.kl_edge_coef = ...

    def forward(self, *args, **kwargs):
        """DDP-safe entry point; loss semantics remain in compute_all_losses()."""
        return self.compute_all_losses(*args, **kwargs)

    def get_reconstruction(self, batch_en, batch_de, batch_g, n_traj_samples=1, run_backwards=True):
        # ---------------------------
        # 1) Encoder (infer z0)
        # ---------------------------
        first_point_mu, first_point_std, encoder_aux = self.encoder_z0(
            batch_en.x,
            batch_en.edge_attr,
            batch_en.edge_index,
            batch_en.pos,
            batch_en.edge_same,
            batch_en.batch,
            batch_en.y,
            return_edge_attn=True,
            return_encoder_aux=True,
        )
        batch_en.event_embeddings = encoder_aux["event_embeddings"]
        batch_en.edge_attn = encoder_aux.get("edge_attn")
        batch_en.physical_edge_labels = batch_g

        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigmas_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0)

        first_point_std = first_point_std.abs()

        time_steps_to_predict = batch_de["time_steps"]

        assert torch.sum(first_point_std < 0) == 0.0
        assert not torch.isnan(time_steps_to_predict).any()
        assert not torch.isnan(first_point_enc).any()

        # ---------------------------
        # 2) Build amortized posterior q(nu_t | obs) and inject into ODE
        # ---------------------------
        edge_extras = None
        if hasattr(self, "edge_posterior") and (self.edge_posterior is not None):
            # Build a time-dependent relation provider from the selected source.
            rel_type_provider, edge_extras = self.edge_posterior.build(
                batch_en,
                time_steps_to_predict,
                sample=False,  # use soft probs for stability first
            )
            # Inject into the ODE function used by odeint
            self.diffeq_solver.ode_func.set_edge_posterior(
                rel_type_provider,
                adjoint_tensors=edge_extras.get("_adjoint_tensors", ()),
            )

            if edge_extras is not None:
                print("edge posterior OK, q_probs_grid:", edge_extras["q_probs_grid"].shape)

        # ---------------------------
        # 3) ODE solve
        # ---------------------------
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict, batch_g)

        if edge_extras is not None:
            message_norms = [
                layer.base_conv.last_message_norm
                for layer in self.diffeq_solver.ode_func.ode_func_net.gcs
                if hasattr(layer.base_conv, "last_message_norm")
            ]
            if message_norms:
                edge_extras["aggregate_nri_message_norm"] = torch.stack(
                    message_norms
                ).mean()
            if hasattr(self.diffeq_solver.ode_func, "last_derivative_norm"):
                edge_extras["latent_derivative_norm"] = (
                    self.diffeq_solver.ode_func.last_derivative_norm
                )

        # ---------------------------
        # 4) Decoder
        # ---------------------------
        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (
                torch.unsqueeze(first_point_mu, 0),
                torch.unsqueeze(first_point_std, 0),
                first_point_enc,
            ),
            "latent_traj": sol_y.detach(),
            "edge_extras": edge_extras,
        }

        return pred_x, all_extra_info, None

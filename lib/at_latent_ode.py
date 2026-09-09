from lib.base_models import VAE_Baseline
import lib.utils as utils
import torch


class ATLatentGraphODE(VAE_Baseline):
    '''
    AT-LG-ODE: identical to LatentGraphODE (lib/latent_ode.py) except that, right after the
    encoder runs, the encoder's last-layer relational attention is read off and cached into
    the shared AttentionTransport module, which the ODE function then reads at every solver
    step to compute time-dependent edge weights w_ij(t).
    '''

    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, transport,
                 z0_prior, device, obsrv_std=None):

        super(ATLatentGraphODE, self).__init__(
            input_dim=input_dim, latent_dim=latent_dim,
            z0_prior=z0_prior,
            device=device, obsrv_std=obsrv_std)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.transport = transport
        self.latent_dim = latent_dim

    def get_reconstruction(self, batch_en, batch_de, batch_g, n_traj_samples=1, run_backwards=True):

        # Encoder (Stages 1-4, unchanged):
        first_point_mu, first_point_std = self.encoder_z0(batch_en.x, batch_en.edge_attr,
                                                            batch_en.edge_index, batch_en.pos, batch_en.edge_same,
                                                            batch_en.batch, batch_en.y)

        # Stage 5 setup: cache the encoder's relational attention for the ODE to transport.
        self.transport.reset()
        attention, edge_index = self.encoder_z0.get_last_attention()
        if attention is not None:
            self.transport.build_cache(attention, edge_index, batch_en.object_id, batch_en.t_s, batch_en.batch)

        means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
        sigmas_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
        first_point_enc = utils.sample_standard_gaussian(means_z0, sigmas_z0)

        first_point_std = first_point_std.abs()

        time_steps_to_predict = batch_de["time_steps"]

        assert (torch.sum(first_point_std < 0) == 0.)
        assert (not torch.isnan(time_steps_to_predict).any())
        assert (not torch.isnan(first_point_enc).any())

        # ODE (Stage 5, attention-weighted) + Decoder (unchanged):
        sol_y = self.diffeq_solver(first_point_enc, time_steps_to_predict, batch_g)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (torch.unsqueeze(first_point_mu, 0), torch.unsqueeze(first_point_std, 0), first_point_enc),
            "latent_traj": sol_y.detach()
        }

        return pred_x, all_extra_info, None

from lib.at_gnn_models import ATEncoderGNN, ATOdeGNN
from lib.at_latent_ode import ATLatentGraphODE
from lib.at_diffeq_solver import ATDiffeqSolver, ATGraphODEFunc
from lib.attention_transport import AttentionTransport
from lib.encoder_decoder import Decoder


def create_AT_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device):

    latent_dim = args.latents
    rec_dim = args.rec_dims
    ode_dim = args.ode_dims

    # Encoder (Stages 1-4, unchanged aside from stashing relational attention)
    encoder_z0 = ATEncoderGNN(in_dim=input_dim, n_hid=rec_dim, out_dim=latent_dim, n_heads=args.n_heads,
                               n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,
                               aggregate=args.rec_attention).to(device)

    if args.augment_dim > 0:
        ode_input_dim = latent_dim + args.augment_dim
    else:
        ode_input_dim = latent_dim

    transport = AttentionTransport(num_atoms=args.n_balls, lam_init=args.lambda_init,
                                    learnable_lambda=args.learnable_lambda,
                                    nonadj_floor=args.nonadj_floor,
                                    ablation=args.ablation).to(device)

    ode_func_net = ATOdeGNN(in_dim=ode_input_dim, n_hid=ode_dim, out_dim=ode_input_dim,
                             n_layers=args.gen_layers, dropout=args.dropout, transport=transport).to(device)

    gen_ode_func = ATGraphODEFunc(ode_func_net=ode_func_net, transport=transport, device=device).to(device)

    diffeq_solver = ATDiffeqSolver(gen_ode_func, args.solver, args=args,
                                    odeint_rtol=1e-2, odeint_atol=1e-2, device=device)

    decoder = Decoder(latent_dim, input_dim).to(device)

    model = ATLatentGraphODE(
        input_dim=input_dim,
        latent_dim=args.latents,
        encoder_z0=encoder_z0,
        decoder=decoder,
        diffeq_solver=diffeq_solver,
        transport=transport,
        z0_prior=z0_prior,
        device=device,
        obsrv_std=obsrv_std,
    ).to(device)

    return model

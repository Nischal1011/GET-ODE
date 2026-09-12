'''
Assembles Edge-GNN exactly like LG-ODE (lib/create_latent_ode_model.py) -- same GraphODEFunc,
DiffeqSolver, Decoder, and LatentGraphODE, all imported unchanged -- swapping only the encoder
for EdgeGNNEncoder (lib/edge_gnn_models.py). LatentGraphODE only ever calls
`self.encoder_z0(x, edge_attr, edge_index, pos, edge_same, batch, y)` and expects (mean, std)
back, so any encoder with that interface plugs in directly.
'''
from lib.gnn_models import GNN
from lib.edge_gnn_models import EdgeGNNEncoder
from lib.latent_ode import LatentGraphODE
from lib.encoder_decoder import Decoder
from lib.diffeq_solver import DiffeqSolver, GraphODEFunc


def create_EdgeGNN_model(args, input_dim, z0_prior, obsrv_std, device):
    latent_dim = args.latents
    rec_dim = args.rec_dims
    ode_dim = args.ode_dims

    encoder_z0 = EdgeGNNEncoder(in_dim=input_dim, n_hid=rec_dim, out_dim=latent_dim,
                                n_layers=args.rec_layers, dropout=args.dropout, aggregate="add").to(device)

    if args.augment_dim > 0:
        ode_input_dim = latent_dim + args.augment_dim
    else:
        ode_input_dim = latent_dim

    ode_func_net = GNN(in_dim=ode_input_dim, n_hid=ode_dim, out_dim=ode_input_dim, n_heads=args.n_heads,
                       n_layers=args.gen_layers, dropout=args.dropout, conv_name="NRI", aggregate="add")

    gen_ode_func = GraphODEFunc(ode_func_net=ode_func_net, device=device).to(device)
    diffeq_solver = DiffeqSolver(gen_ode_func, args.solver, args=args, odeint_rtol=1e-2, odeint_atol=1e-2, device=device)

    decoder = Decoder(latent_dim, input_dim).to(device)

    model = LatentGraphODE(
        input_dim=input_dim,
        latent_dim=args.latents,
        encoder_z0=encoder_z0,
        decoder=decoder,
        diffeq_solver=diffeq_solver,
        z0_prior=z0_prior,
        device=device,
        obsrv_std=obsrv_std,
    ).to(device)

    return model

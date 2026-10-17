from lib.gnn_models import GNN
from lib.latent_ode import LatentGraphODE
from lib.encoder_decoder import Decoder
from lib.diffeq_solver import DiffeqSolver, GraphODEFunc
from lib.at_transport_posterior import ATTransportPosterior
from lib.evidence_transport_posterior import (
    EvidenceTransportPosterior,
    RelationalEvidenceHead,
    StaticRelationPosterior,
)


DATASET_ALIASES = {
    "dataset1": "spring",
    "dataset2": "charged",
    "dataset3": "motion",
    "dataset4": "pems08",
}


RELATION_SEMANTICS = {
    "spring": {
        "labels": ("no spring", "spring"),
        "inactive_relation_index": 0,
    },
    "charged": {
        "labels": ("opposite charges", "like charges"),
        "inactive_relation_index": None,
    },
    "motion": {
        "labels": ("no structural link", "structural link"),
        "inactive_relation_index": 0,
    },
    "pems08": {
        "labels": ("no road link", "road link"),
        "inactive_relation_index": 0,
    },
}


def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device):
    # -------------------------
    # dimensions
    # -------------------------
    latent_dim = args.latents
    rec_dim = args.rec_dims
    ode_dim = args.ode_dims

    # -------------------------
    # encoder (z0)
    # -------------------------
    encoder_z0 = GNN(
        in_dim=input_dim,
        n_hid=rec_dim,
        out_dim=latent_dim,
        n_heads=args.n_heads,
        n_layers=args.rec_layers,
        dropout=args.dropout,
        conv_name=args.z0_encoder,
        aggregate=args.rec_attention,
    )

    # -------------------------
    # ODE function net
    # -------------------------
    ode_input_dim = latent_dim + args.augment_dim if args.augment_dim > 0 else latent_dim

    ode_func_net = GNN(
        in_dim=ode_input_dim,
        n_hid=ode_dim,
        out_dim=ode_input_dim,
        n_heads=args.n_heads,
        n_layers=args.gen_layers,
        dropout=args.dropout,
        conv_name=args.odenet,
        aggregate="add",
    )

    gen_ode_func = GraphODEFunc(ode_func_net=ode_func_net, device=device).to(device)

    diffeq_solver = DiffeqSolver(
        gen_ode_func,
        args.solver,
        args=args,
        odeint_rtol=1e-2,
        odeint_atol=1e-2,
        device=device,
    )

    # -------------------------
    # decoder
    # -------------------------
    decoder = Decoder(latent_dim, input_dim).to(device)

    # -------------------------
    # model
    # -------------------------
    model = LatentGraphODE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_z0=encoder_z0,
        decoder=decoder,
        diffeq_solver=diffeq_solver,
        z0_prior=z0_prior,
        device=device,
        obsrv_std=obsrv_std,
    ).to(device)

    # -------------------------
    # Edge posterior. Attention remains the backward-compatible default.
    # -------------------------
    posterior_source = getattr(args, "edge_posterior_source", "attention")
    prior_mode = getattr(args, "edge_prior_mode", "uniform")
    edge_prior_p = getattr(args, "edge_prior_p", 0.9)
    prior_strength = getattr(args, "edge_prior_strength", 1.0)
    tau_gumbel = getattr(args, "gumbel_tau", 1.0)
    hard_gumbel = getattr(args, "gumbel_hard", False)

    if posterior_source in {"attention", "constant"}:
        model.edge_posterior = ATTransportPosterior(
            n_atoms=args.n_balls,
            edge_types=args.edge_types,
            K_lag=getattr(args, "K_lag", 16),
            L=1.0,
            v=getattr(args, "attn_v", 1.0),
            tau_gumbel=tau_gumbel,
            hard_gumbel=hard_gumbel,
            source_mode=posterior_source,
        ).to(device)
    elif posterior_source == "evidence":
        evidence_head = RelationalEvidenceHead(
            event_dim=rec_dim,
            edge_types=args.edge_types,
            hidden_dim=getattr(args, "evidence_hidden_dim", 64),
            init_bias=getattr(args, "evidence_init_bias", -2.0),
        )
        model.edge_posterior = EvidenceTransportPosterior(
            n_atoms=args.n_balls,
            edge_types=args.edge_types,
            rel_send=diffeq_solver.rel_send,
            rel_rec=diffeq_solver.rel_rec,
            evidence_head=evidence_head,
            K_lag=getattr(args, "K_lag", 16),
            lag_max=getattr(args, "lag_max", 1.0),
            transport_velocity=getattr(args, "attn_v", 1.0),
            decay_init=getattr(args, "evidence_decay_init", 1.0),
            age_kernel=getattr(args, "age_kernel", "uniform"),
            prior_mode=prior_mode,
            edge_prior_p=edge_prior_p,
            prior_strength=prior_strength,
            tau_gumbel=tau_gumbel,
            hard_gumbel=hard_gumbel,
            ablation=getattr(args, "evidence_ablation", "full"),
        ).to(device)
        evidence_ablation = getattr(args, "evidence_ablation", "full")
        if evidence_ablation == "prior-only":
            for parameter in model.edge_posterior.parameters():
                parameter.requires_grad_(False)
        if evidence_ablation == "instantaneous":
            model.edge_posterior.raw_decay_rates.requires_grad_(False)
    elif posterior_source == "topology":
        model.edge_posterior = StaticRelationPosterior(
            n_atoms=args.n_balls,
            edge_types=args.edge_types,
            rel_send=diffeq_solver.rel_send,
            rel_rec=diffeq_solver.rel_rec,
            prior_mode=prior_mode,
            edge_prior_p=edge_prior_p,
            prior_strength=prior_strength,
            tau_gumbel=tau_gumbel,
            hard_gumbel=hard_gumbel,
        ).to(device)
    elif posterior_source == "none":
        model.edge_posterior = None
    else:
        raise ValueError(f"Unknown edge posterior source: {posterior_source}")

    resolved_data = DATASET_ALIASES.get(getattr(args, "data", "spring"), getattr(args, "data", "spring"))
    relation_semantics = RELATION_SEMANTICS.get(
        resolved_data,
        {"labels": tuple(str(i) for i in range(args.edge_types)), "inactive_relation_index": None},
    )
    inactive_relation_index = (
        relation_semantics["inactive_relation_index"]
        if posterior_source == "evidence"
        else None
    )
    for layer in ode_func_net.gcs:
        if hasattr(layer.base_conv, "inactive_relation_index"):
            layer.base_conv.inactive_relation_index = inactive_relation_index
            layer.base_conv.skip_first_edge_type = inactive_relation_index == 0
            if inactive_relation_index is not None:
                for parameter in layer.base_conv.msg_fc1[inactive_relation_index].parameters():
                    parameter.requires_grad_(False)
                for parameter in layer.base_conv.msg_fc2[inactive_relation_index].parameters():
                    parameter.requires_grad_(False)

    # store prior config for KL(q(nu)||p(nu)) in base_models.py
    model.edge_prior_mode = prior_mode
    model.edge_prior_p = edge_prior_p
    model.edge_prior_strength = prior_strength
    model.kl_edge_coef = getattr(args, "kl_edge_coef", 1.0)
    model.evidence_l1_coef = getattr(args, "evidence_l1_coef", 0.0)
    model.edge_posterior_source = posterior_source
    model.relation_semantics = relation_semantics

    return model

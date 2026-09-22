import os
import sys
from lib.new_dataLoader import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.create_latent_ode_model import create_LatentODE_model
from lib.utils import compute_loss_all_batches
from lib.debug_snap import plot_quick_stage

# Generative model for noisy raw based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of objects in the dataset.')
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr',  type=float, default=5e-4, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--save-graph', type=str, default='plot/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--data', type=str, default='spring', help="spring,charged,motion")
parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('-l', '--latents', type=int, default=16, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default=128, help="Dimensionality of the ODE func")
parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in recognition model ")
parser.add_argument('--n-heads', type=int, default=1, help="Number of heads in GTrans")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")
parser.add_argument('--extrap', type=str,default="False", help="Set extrapolation mode. If this flag is not set, run interpolation mode.")
parser.add_argument('--dropout', type=float, default=0.2,help='Dropout rate (1 - keep probability).')
parser.add_argument('--sample-percent-train', type=float, default=0.4,help='Percentage of training observtaion raw')
parser.add_argument('--sample-percent-test', type=float, default=0.4,help='Percentage of testing observtaion raw')
parser.add_argument('--augment_dim', type=int, default=128, help='augmented dimension')
parser.add_argument('--edge_types', type=int, default=2, help='edge number in NRI')
parser.add_argument('--odenet', type=str, default="NRI", help='NRI')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')
parser.add_argument('--l2', type=float, default=1e-3, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--cutting_edge', type=bool, default=True, help='True/False')
parser.add_argument('--extrap_num', type=int, default=40, help='extrap num ')
parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--alias', type=str, default="run")



# -------------------------------------------------
# AT-ODE: Attention Transport parameters
# -------------------------------------------------

parser.add_argument(
    '--K_lag',
    type=int,
    default=16,
    help='Length of the lag dimension for attention transport. '
         'Controls how long inferred edge relevance can persist over time. '
         'K_lag=0 reduces AT-ODE to instantaneous attention (LG-ODE behavior).'
)

parser.add_argument(
    '--attn_v',
    type=float,
    default=1.0,
    help='Transport velocity along the normalized evidence-age axis.'
)

parser.add_argument(
    '--gumbel_tau',
    type=float,
    default=1.0,
    help='Temperature parameter for Gumbel-Softmax sampling of edge variables. '
         'Higher values yield smoother, more uncertain edge posteriors; '
         'lower values approximate discrete edge selection.'
)

parser.add_argument(
    '--gumbel_hard',
    type=bool,
    default=False,
    help='If True, uses straight-through Gumbel-Softmax to sample discrete edges. '
         'If False, uses soft probabilistic edges (recommended for stability).'
)

parser.add_argument(
    '--edge_prior_mode',
    type=str,
    default='graph',
    choices=['uniform', 'graph'],
    help='Form of the prior over edge variables. '
         '"uniform" assumes no structural preference among edges; '
         '"graph" biases the prior toward known or observed graph structure.'
)

parser.add_argument(
    '--edge_prior_p',
    type=float,
    default=0.9,
    help='Prior probability assigned to structurally valid edges when '
         'edge_prior_mode="graph". Higher values enforce stronger structural bias.'
)

parser.add_argument(
    '--kl_edge_coef',
    type=float,
    default=1.0,
    help='Weight of the KL-divergence term between the inferred edge posterior '
         'and the edge prior. Controls the tradeoff between data-driven edge '
         'inference and prior structural regularization.'
)

parser.add_argument(
    '--edge-posterior-source', '--edge_posterior_source',
    dest='edge_posterior_source',
    choices=['none', 'attention', 'evidence', 'constant', 'topology'],
    default='attention',
    help='Source for generator relation probabilities; attention preserves the existing AT path.'
)
parser.add_argument(
    '--evidence-hidden-dim', '--evidence_hidden_dim',
    dest='evidence_hidden_dim', type=int, default=64,
    help='Hidden width of the temporal-edge relational evidence MLP.'
)
parser.add_argument(
    '--evidence-init-bias', '--evidence_init_bias',
    dest='evidence_init_bias', type=float, default=-2.0,
    help='Initial evidence-head output bias; negative values keep the initial posterior near its prior.'
)
parser.add_argument(
    '--evidence-decay-init', '--evidence_decay_init',
    dest='evidence_decay_init', type=float, default=1.0,
    help='Initial nonnegative stale-evidence decay rate for every relation type.'
)
parser.add_argument(
    '--edge-prior-strength', '--edge_prior_strength',
    dest='edge_prior_strength', type=float, default=1.0,
    help='Positive concentration multiplying the categorical physical-edge prior.'
)
parser.add_argument(
    '--age-kernel', '--age_kernel',
    dest='age_kernel', choices=['uniform', 'exponential', 'learned'], default='uniform',
    help='Lag-bin readout kernel; uniform is the controlled-comparison default.'
)
parser.add_argument(
    '--lag-max', '--lag_max',
    dest='lag_max', type=float, default=1.0,
    help='Oldest retained evidence age in normalized time units.'
)
parser.add_argument(
    '--evidence-l1-coef', '--evidence_l1_coef',
    dest='evidence_l1_coef', type=float, default=0.0,
    help='Optional mean-evidence penalty; zero leaves the comparison loss unchanged.'
)
parser.add_argument(
    '--evidence-ablation', '--evidence_ablation',
    dest='evidence_ablation',
    choices=['full', 'instantaneous', 'shuffle-times', 'shuffle-values', 'prior-only'],
    default='full',
    help='Controlled evidence-mode ablation applied before age transport.'
)





args = parser.parse_args()
assert(int(args.rec_dims%args.n_heads) ==0)
utils.set_global_args(args)



if args.data == "spring":
    args.dataset = 'data/spring'
    args.suffix = '_springs5'
    args.total_ode_step=60
elif args.data == "charged":
    args.dataset = 'data/charged'
    args.suffix = '_charged5'
    args.total_ode_step=60
elif args.data == "motion":
    args.dataset = 'data/motion'
    args.suffix = '_mocap35'
    args.total_ode_step=49
    args.n_balls = 31



############ CPU AND GPU related, Mode related, Dataset Related
if torch.cuda.is_available():
	print("Using GPU" + "-"*80)
	device = torch.device("cuda:0")
else:
	print("Using CPU" + "-" * 80)
	device = torch.device("cpu")

if args.extrap == "True":
    print("Running extrap mode" + "-"*80)
    args.mode = "extrap"
elif args.extrap=="False":
    print("Running interp mode" + "-" * 80)
    args.mode="interp"



def do_eval_and_plots(model, enc_batch, dec_batch, graph_batch, pred_samples, device, args, epoch=None):
    """
    Plot using the EXACT tensors you pass in (same batch the loss used).
    No dataloader iteration or extra forward happens here.

    enc_batch  : dict or PyG Batch/Data (encoder batch)  [used only for meta if needed]
    dec_batch  : dict or PyG Batch/Data (decoder batch)  MUST contain time_steps, raw, mask, (optional) types
    graph_batch: dict or PyG Batch/Data (graph batch)     used for structure/meta/animations
    pred_samples: torch.Tensor or np.ndarray of shape [S, M, T, D] from your loss step
    """
    import os, json, numpy as np, torch
    from lib import plots
    import lib.utils as utils
    from torch_geometric.data import Batch, Data

    # ---------- small helpers ----------
    def _to_device(x):
        if isinstance(x, dict):
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in x.items()}
        if isinstance(x, (Batch, Data)):
            return x.to(device)
        return x  # assume already on device / CPU arrays

    def _np(x):
        import numpy as np
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()

        # handle lists/tuples of tensors or arrays
        if isinstance(x, (list, tuple)):
            return np.array([_np(e) for e in x], dtype=object if any(
                not isinstance(e, (np.ndarray, float, int)) for e in x) else None)

        # handle dicts (optional)
        if isinstance(x, dict):
            return {k: _np(v) for k, v in x.items()}

        return np.asarray(x)


    def _get(b, k, default=None):
        try: return b[k]
        except Exception: return getattr(b, k, default)
    def _sanitize(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)

    # ---------- io ----------
    out_dir = os.path.join(args.save_graph, f"epoch_{epoch:04d}") if epoch is not None else args.save_graph
    os.makedirs(out_dir, exist_ok=True)

    # ---------- use passed batches as-is (NO get_next_batch_new) ----------
    batch_enc   = _to_device(enc_batch)
    batch_dec   = _to_device(dec_batch)
    batch_graph = _to_device(graph_batch)

    # ---------- decoder tensors used in LOSS ----------
    t_grid = _get(batch_dec, "time_steps")           # [T]
    y_all  = _get(batch_dec, "raw")                 # [M,T,D]
    mk_all = _get(batch_dec, "mask")                 # [M,T,D]
    types_m = _get(batch_dec, "types", None)         # [M] or None
    if isinstance(y_all, torch.Tensor): y_all = y_all.detach()
    if isinstance(mk_all, torch.Tensor): mk_all = mk_all.detach()
    M, T, D = y_all.shape

    # ---------- predictions: exactly what you passed ----------
    ps = _np(pred_samples)                           # [S,M,T,D]
    assert ps.ndim == 4 and ps.shape[1:] == (M, T, D), \
        f"pred_samples shape {ps.shape} incompatible with [S,M,T,D]=[, {M},{T},{D}]"
    mu_all  = ps.mean(axis=0)                        # [M,T,D]
    sig_ep  = ps.std(axis=0, ddof=1)                 # [M,T,D]
    sig_tot = sig_ep                                 # (plug aleatoric later if you model it)

    # ---------- meta (best-effort) ----------
    type_names = list(getattr(args, "sensor_types", []))
    try:
        with open(getattr(args, "typecols_path", "data/type_cols.json"), "r") as f:
            type_cols_map = {str(k).lower(): [int(i) for i in v] for k, v in json.load(f).items()}
    except Exception:
        type_cols_map = {}
    try:
        with open(getattr(args, "catalog_path", "data/feature_catalog.json"), "r") as f:
            feat_catalog = json.load(f)
        if not isinstance(feat_catalog, list): raise ValueError
    except Exception:
        feat_catalog = [f"feat_{i}" for i in range(D)]
    try:
        with open(getattr(args, "busids_path", "data/bus_ids.json"), "r") as f:
            bus_ids = json.load(f)
    except Exception:
        bus_ids = [str(i) for i in range(M)]

    # time axis for display = exactly what loss used
    t_display = _np(t_grid)

    # ---------- (optional) network animation ----------
    try:
        g_list = batch_graph.to_data_list() if hasattr(batch_graph, "to_data_list") else [batch_graph]
        g0 = g_list[0]
        edge_index_struct = getattr(g0, "edge_index", None)
        node_types_struct = getattr(g0, "sensor_idx", None)
        N0 = int(getattr(g0, "num_nodes", node_types_struct.numel() if node_types_struct is not None else M))
        N_struct = min(N0, M)

        feat = int(getattr(args, "feat_idx", 0))
        feat_label = feat_catalog[feat] if 0 <= feat < len(feat_catalog) else f"feature {feat}"

        y0  = _np(y_all[:N_struct])
        mu0 = _np(mu_all[:N_struct])
        mk0 = _np(mk_all[:N_struct]).astype(bool)

        gif_prefix = os.path.join(out_dir, f"network_feat{feat}")
        plots.animate_network_three(
            edge_index=edge_index_struct,
            t_seconds=t_display,
            truth=y0, mu=mu0, mask=mk0,
            types=(node_types_struct[:N_struct] if node_types_struct is not None else None),
            type_names=type_names,
            feat_idx=feat,
            pos_xy=getattr(g0, "pos_xy", None),
            save_prefix=gif_prefix,
            interval_ms=int(getattr(args, "viz_interval_ms", 800)),
            step=int(getattr(args, "viz_step", 1)),
            obs_mask_override=None,
            feature_name=feat_label,
        )
        print(f"[viz] wrote {gif_prefix}_observed.gif / _predicted.gif / _error.gif")
    except Exception as e:
        print("[viz] network animation failed:", repr(e))

    # ---------- structural snapshot ----------
    try:
        g_list = batch_graph.to_data_list() if hasattr(batch_graph, "to_data_list") else [batch_graph]
        g0 = g_list[0]
        plots.plot_structural_with_ids(
            edge_index=g0.edge_index,
            num_nodes=int(getattr(g0, "num_nodes", g0.sensor_idx.numel())),
            sensor_idx=g0.sensor_idx,
            type_names=type_names,
            savepath=os.path.join(out_dir, "structural_with_ids.png"),
        )
        plots.export_node_types_csv(
            sensor_idx=g0.sensor_idx,
            type_names=type_names,
            csv_path=os.path.join(out_dir, "node_types.csv"),
        )
    except Exception as e:
        print("[viz] static structural grid failed:", repr(e))

    # ---------- node/feature series (loss-exact) ----------
    def _parse_nodes(v, M):
        if v is None: return None
        if isinstance(v, (list, tuple)): return [int(i) for i in v if 0 <= int(i) < M]
        s = str(v).strip()
        if not s: return None
        out = []
        for tok in s.replace(" ", "").split(","):
            if not tok: continue
            try:
                k = int(tok)
                if 0 <= k < M: out.append(k)
            except: pass
        return out or None

    explicit_nodes = _parse_nodes(getattr(args, "nodes_to_plot", None), M)
    nodes_to_plot = explicit_nodes if explicit_nodes is not None else list(range(min(int(getattr(args, "plot_nodes", 6)), M)))

    for m in nodes_to_plot:
        if types_m is not None and len(type_names) > 0:
            t_idx = int(types_m[m].item()) if isinstance(types_m, torch.Tensor) else int(types_m[m])
            tname = type_names[t_idx] if 0 <= t_idx < len(type_names) else None
            feats = type_cols_map.get((tname or "").lower(), [])
        else:
            tname, feats = None, list(range(D))
        if not feats: continue

        bus_id = bus_ids[m] if (0 <= m < len(bus_ids)) else str(m)

        for feat in feats:
            if not (0 <= int(feat) < D): continue
            feat_name = feat_catalog[int(feat)] if int(feat) < len(feat_catalog) else f"feat_{int(feat)}"
            plots.plot_series_with_ci(
                tt=t_grid,
                truth=y_all[m],              # [T,D]
                mu=mu_all[m],                # [T,D]
                sigma=sig_tot[m],            # [T,D]
                mask=mk_all[m],              # [T,D]
                feat_idx=int(feat),
                title=f"Node {m} (bus {bus_id}) • {feat_name} reconstruction ±95% CI",
                savepath=os.path.join(out_dir, f"series_node{m}_bus{_sanitize(str(bus_id))}_{_sanitize(feat_name)}.png"),
                type_name=tname,
                t_display=t_display,
                feature_name=feat_name,
                bus_id=str(bus_id),
                node_idx=int(m),
            )

    # ---------- per-node / per-feature diagnostics ----------
    rmse = plots.per_node_rmse([y_all[i] for i in range(M)],
                               [mu_all[i] for i in range(M)],
                               [mk_all[i] for i in range(M)])
    plots.plot_node_bars(rmse, title="Decoder RMSE per node",
                         savepath=os.path.join(out_dir, "plot_node_bars.png"))

    plots.plot_uncertainty_vs_error(y_all, mu_all, sig_tot, mask=mk_all,
                                    savepath=os.path.join(out_dir, "uncertainty_vs_error.png"))
    lev, cov = plots.coverage_curve(y_all, mu_all, sig_tot, mask=mk_all)
    plots.plot_coverage(lev, cov, savepath=os.path.join(out_dir, "plot_coverage.png"))

    feat_mse, feat_count = plots.mse_per_feature(y_all, mu_all, mask=mk_all)
    plots.plot_feature_bars(
        feat_mse, labels=feat_catalog,
        title="Feature-wise MSE (this test graph)",
        savepath=os.path.join(out_dir, "feature_mse_bars.png")
    )
    try:
        import csv
        with open(os.path.join(out_dir, "feature_mse.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["feature", "mse", "count"])
            for d, name in enumerate(feat_catalog):
                w.writerow([name, float(feat_mse[d]), int(feat_count[d])])
        print(f"[eval] wrote {os.path.join(out_dir, 'feature_mse.csv')}")
    except Exception as e:
        print("[eval] failed to write feature_mse.csv:", repr(e))




#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.autograd.set_detect_anomaly(True)

    ############ Saving Path and Preload.
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    utils.makedirs(args.save_graph)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)


    ############ Loading Data
    print("Loading dataset: " + args.dataset)

    dataloader = ParseData(args.dataset,suffix=args.suffix,mode=args.mode, args =args)


    test_encoder, test_decoder, test_graph, test_batch = dataloader.load_data(sample_percent=args.sample_percent_test,
                                                                              batch_size=args.batch_size,
                                                                              data_type="test")

    train_encoder,train_decoder, train_graph,train_batch = dataloader.load_data(sample_percent=args.sample_percent_train,batch_size=args.batch_size,data_type="train")



    input_dim = dataloader.feature

    ############ Command Related
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    ############ Model Select
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)


    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)
        #exit()

    ##################################################################
    # Training

    log_path = "logs/" + args.alias +"_" + args.z0_encoder+ "_" + args.data + "_" +str(args.sample_percent_train)+ "_" + args.mode + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)


    wait_until_kl_inc = 10
    best_test_mse = np.inf
    n_iters_to_viz = 1

    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef):

        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                             n_traj_samples=3, kl_coef=kl_coef)

        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res["mse"],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"]

    def train_epoch(epo):
        model.train()
        loss_list = []
        mse_list = []
        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []

        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):

            #utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)
            wait_until_kl_inc = 10

            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)

            batch_dict_graph = utils.get_next_batch_new(train_graph, device)

            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, mse,likelihood,kl_first_p,std_first_p = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef)
        
            #saving results
            loss_list.append(loss), mse_list.append(mse), likelihood_list.append(
               likelihood)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)

            # snap_path = plot_quick_stage(
            #     model=model,
            #     batch_enc=batch_dict_encoder,  # PyG Batch
            #     batch_dec=batch_dict_decoder,  # dict from your collate
            #     batch_graph=batch_dict_graph,  # PyG Batch (structural)
            #     node_idx=0,
            #     feat_idx=0,
            #     latent_dim_to_plot=0,
            #     sample_id=0,
            #     total_ode_step=args.total_ode_step,
            #     mode=getattr(args, "mode", "interp"),
            #     tag="raw",
            #     epoch=epo,
            #     step=itr,
            #     outdir="train_snaps"
            # )
            # print("snapshot:", snap_path)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()


        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            epo,
            np.mean(loss_list), np.mean(mse_list), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list))


        return message_train,kl_coef


    for epo in range(1, args.niters + 1):

        message_train, kl_coef = train_epoch(epo)

        if epo % n_iters_to_viz == 0:
            model.eval()
            test_res,batch_dict_encoder, batch_dict_graph, batch_dict_decoder, results  = compute_loss_all_batches(model, test_encoder, test_graph, test_decoder,
                                                n_batches=test_batch, device=device,
                                                n_traj_samples=3, kl_coef=kl_coef)

            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                epo,
                test_res["loss"], test_res["mse"], test_res["likelihood"],
                test_res["kl_first_p"], test_res["std_first_p"])

            # do_eval_and_plots(model,
            #                   batch_dict_encoder, batch_dict_decoder, batch_dict_graph, results["pred_samples"][-1],
            #                   # last batch
            #                   device, args, epoch=epo)

            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_test)
            logger.info("KL coef: {}".format(kl_coef))
            print("raw: %s, encoder: %s, sample: %s, mode:%s" % (
                args.data, args.z0_encoder, str(args.sample_percent_train), args.mode))

            if test_res["mse"] < best_test_mse:
                best_test_mse = test_res["mse"]
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best mse {:.6f}|'.format(epo,
                                                                                                        best_test_mse)
                logger.info(message_best)
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.z0_encoder + "_" + args.data + "_" + str(
                    args.sample_percent_train) + "_" + args.mode + "_epoch_" + str(epo) + "_mse_" + str(
                    best_test_mse) + '.ckpt')
                utils.save_checkpoint_pf(
                    ckpt_path,
                    model, optimizer, scheduler,
                    epoch=epo,
                    best_val_loss=best_test_mse,
                    args=args,
                    device=device
                )


            torch.cuda.empty_cache()














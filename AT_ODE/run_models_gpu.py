import os
import sys
import argparse
import numpy as np
from random import SystemRandom
from datetime import timedelta

# PyTorch can transparently run unsupported MPS operations on the CPU instead of
# aborting the entire training run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributions.normal import Normal
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

import lib.utils as utils
from lib.new_dataLoader import ParseData
from lib.create_latent_ode_model import create_LatentODE_model
from lib.utils import compute_loss_all_batches
from lib.debug_snap import plot_quick_stage


# -------------------------------------------------
# Runtime / device setup (KEEP AS REQUESTED)
# -------------------------------------------------

def setup_runtime(requested_device="auto"):
    if "PMI_RANK" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Polaris execution requires CUDA GPUs")
        global_rank = int(os.environ["PMI_RANK"])
        world_size = int(os.environ["PMI_SIZE"])
        local_rank = global_rank % torch.cuda.device_count()

        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            world_size=world_size,
        )

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_dist = True
    else:
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if requested_device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif mps_available:
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
            device = torch.device("cuda")
        elif requested_device == "mps":
            if not mps_available:
                raise RuntimeError("--device mps was requested, but Apple MPS is unavailable")
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        global_rank = 0
        local_rank = 0
        world_size = 1
        is_dist = False

    return device, is_dist, local_rank, global_rank, world_size


# -------------------------------------------------
# Argument parsing (UNCHANGED)
# -------------------------------------------------

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--n-balls', type=int, default=5)
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--save', type=str, default='experiments/')
parser.add_argument('--save-graph', type=str, default='plot/')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('-r', '--random-seed', type=int, default=1991)
parser.add_argument('--data', type=str, default='spring')
parser.add_argument('--z0-encoder', type=str, default='GTrans')
parser.add_argument('-l', '--latents', type=int, default=16)
parser.add_argument('--rec-dims', type=int, default=64)
parser.add_argument('--ode-dims', type=int, default=128)
parser.add_argument('--rec-layers', type=int, default=2)
parser.add_argument('--n-heads', type=int, default=1)
parser.add_argument('--gen-layers', type=int, default=1)
parser.add_argument('--extrap', type=str, default="False")
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--sample-percent-train', type=float, default=0.4)
parser.add_argument('--sample-percent-test', type=float, default=0.4)
parser.add_argument('--augment_dim', type=int, default=128)
parser.add_argument('--edge_types', type=int, default=2)
parser.add_argument('--odenet', type=str, default="NRI")
parser.add_argument('--solver', type=str, default="rk4")
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default="AdamW")
parser.add_argument('--clip', type=float, default=10)
parser.add_argument('--cutting_edge', type=bool, default=True)
parser.add_argument('--extrap_num', type=int, default=40)
parser.add_argument('--rec_attention', type=str, default="attention")
parser.add_argument('--alias', type=str, default="run")
parser.add_argument(
    '--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
    help='Compute device. auto prefers CUDA, then Apple MPS, then CPU.'
)
parser.add_argument(
    '--validation-fraction', type=float, default=0.1,
    help='Fraction of training graphs reserved for deterministic validation.'
)
parser.add_argument(
    '--early-stopping-patience', type=int, default=15,
    help='Stop after this many validation checks without sufficient improvement; 0 disables stopping.'
)
parser.add_argument(
    '--early-stopping-min-delta', type=float, default=1e-5,
    help='Minimum validation-MSE decrease counted as an improvement.'
)


parser.add_argument(
    '--amp',
    action='store_true',
    help='Enable automatic mixed precision (AMP) training on GPUs.'
)

# AT-ODE params (UNCHANGED)
parser.add_argument('--K_lag', type=int, default=16)
parser.add_argument(
    '--attn_v',
    type=float,
    default=1.0,
    help='Transport velocity along the normalized evidence-age axis.'
)
parser.add_argument('--gumbel_tau', type=float, default=1.0)
parser.add_argument('--gumbel_hard', type=bool, default=False)
parser.add_argument('--edge_prior_mode', type=str, default="graph")
parser.add_argument('--edge_prior_p', type=float, default=0.9)
parser.add_argument('--kl_edge_coef', type=float, default=1.0)
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
assert int(args.rec_dims % args.n_heads) == 0
if not 0.0 < args.validation_fraction < 1.0:
    parser.error('--validation-fraction must be between 0 and 1')
if args.early_stopping_patience < 0:
    parser.error('--early-stopping-patience must be nonnegative')
if args.early_stopping_min_delta < 0:
    parser.error('--early-stopping-min-delta must be nonnegative')
utils.set_global_args(args)

DATASET_ALIASES = {
    "dataset1": "spring",
    "dataset2": "charged",
    "dataset3": "motion",
    "dataset4": "pems08",
}

args.data_label = args.data
resolved_data = DATASET_ALIASES.get(args.data, args.data)





# -------------------------------------------------
# Dataset config (UNCHANGED)
# -------------------------------------------------

if resolved_data == "spring":
    args.dataset = 'data/spring'
    args.suffix = '_springs5'
    args.total_ode_step = 60
elif resolved_data == "charged":
    args.dataset = 'data/charged'
    args.suffix = '_charged5'
    args.total_ode_step = 60
elif resolved_data == "motion":
    args.dataset = 'data/motion'
    args.suffix = '_mocap35'
    args.total_ode_step = 49
    args.n_balls = 31
elif resolved_data == "pems08":
    args.dataset = 'data/pems08'
    args.suffix = '_pems08'
    args.total_ode_step = 60
    args.n_balls = 170
else:
    raise ValueError(f"Unknown dataset label: {args.data}")


if args.extrap == "True":
    print("Running extrap mode" + "-"*80)
    args.mode = "extrap"
elif args.extrap=="False":
    print("Running interp mode" + "-" * 80)
    args.mode="interp"




# -------------------------------------------------
# Runtime init (NEW, SAFE)
# -------------------------------------------------

device, is_dist, local_rank, global_rank, world_size = setup_runtime(args.device)
use_amp = args.amp and device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == '__main__':

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    utils.makedirs(args.save)
    utils.makedirs(args.save_graph)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    print("Loading dataset label:", args.data_label)

    dataloader = ParseData(args.dataset, suffix=args.suffix, mode=args.mode, args=args)

    train_encoder, train_decoder, train_graph, train_batch = dataloader.load_data(
        sample_percent=args.sample_percent_train,
        batch_size=args.batch_size,
        data_type="train"
    )

    val_encoder, val_decoder, val_graph, val_batch = dataloader.load_data(
        sample_percent=args.sample_percent_train,
        batch_size=args.batch_size,
        data_type="val"
    )

    test_encoder, test_decoder, test_graph, test_batch = dataloader.load_data(
        sample_percent=args.sample_percent_test,
        batch_size=args.batch_size,
        data_type="test"
    )

    input_dim = dataloader.feature

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    obsrv_std = torch.tensor([0.01], device=device)
    z0_prior = Normal(torch.tensor([0.0], device=device),
                      torch.tensor([1.0], device=device))

    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device)

    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None
        )

    print("Model device:", next(model.parameters()).device)

    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 1000, eta_min=1e-9
    )

    # -------------------------------------------------
    # Resume from checkpoint (FULL resume)
    # -------------------------------------------------
    start_epoch = 1
    best_val_mse = np.inf
    best_epoch = None
    best_model_state = None
    best_checkpoint_path = None
    epochs_without_improvement = 0
    n_iters_to_viz = 1


    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")

        start_epoch, best_val_mse, resumed_checkpoint = utils.resume_from_pf(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            device=device,
            return_checkpoint=True,
        )
        resumed_args = resumed_checkpoint.get("args", {})
        has_validation_metadata = (
            "validation_fraction" in resumed_args
            if isinstance(resumed_args, dict)
            else hasattr(resumed_args, "validation_fraction")
        )
        if not has_validation_metadata:
            print("[INFO] Legacy checkpoint detected; resetting the validation baseline.")
            best_val_mse = np.inf
        best_epoch = start_epoch - 1
        model_for_state = model.module if hasattr(model, "module") else model
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model_for_state.state_dict().items()
        }
        best_checkpoint_path = ckpt_path

        print(
            f"[INFO] Resumed at epoch {start_epoch}")



    # scaler = GradScaler(enabled=(device.type == "cuda"))


    log_path = "logs/" + args.alias +"_" + args.z0_encoder+ "_" + args.data_label + "_" +str(args.sample_percent_train)+ "_" + args.mode + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")

    if global_rank == 0:
        logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
        logger.info(input_command)
        logger.info(str(args))
        logger.info(args.alias)

        wandb_run = None
        if wandb is not None:
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "at_ode"),
                entity=os.getenv("WANDB_ENTITY") or None,
                name=f"{args.alias}_{experimentID}",
                config=vars(args),
                mode=os.getenv("WANDB_MODE", "disabled"),
                reinit=True,
            )
            if wandb_run is not None:
                wandb_run.config.update(
                    {"experiment_id": experimentID, "input_command": input_command},
                    allow_val_change=True,
                )
    else:
        wandb_run = None
        logger = None

    # -------------------------------------------------
    # TRAINING (LOGIC UNCHANGED, AMP ONLY)
    # -------------------------------------------------

    def train_single_batch(model, batch_dict_encoder, batch_dict_decoder, batch_dict_graph, kl_coef):

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            train_res = model(
                batch_dict_encoder,
                batch_dict_decoder,
                batch_dict_graph,
                n_traj_samples=3,
                kl_coef=kl_coef
            )
            loss = train_res["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.detach().item()

        del loss
        # torch.cuda.empty_cache()

        return (
            loss_value,
            train_res["mse"],
            train_res["likelihood"],
            train_res["kl_first_p"],
            train_res["std_first_p"],
        )


    def train_epoch(epo):
        model.train()
        loss_list, mse_list, ll_list, kl_list, std_list = [], [], [], [], []

        iterator = range(train_batch)
        if global_rank == 0:
            iterator = tqdm(iterator)

        for itr in iterator:

            if itr < 10:
                kl_coef = 0.0
            else:
                kl_coef = (1 - 0.99 ** (itr - 10))

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_graph   = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, mse, ll, kl, std = train_single_batch(
                model,
                batch_dict_encoder,
                batch_dict_decoder,
                batch_dict_graph,
                kl_coef
            )

            loss_list.append(loss)
            mse_list.append(mse)
            ll_list.append(ll)
            kl_list.append(kl)
            std_list.append(std)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder

        scheduler.step()

        mean_loss = float(np.mean(loss_list))
        mean_mse = float(np.mean(mse_list))
        mean_ll = float(np.mean(ll_list))
        mean_kl = float(np.mean(kl_list))
        mean_std = float(np.mean(std_list))

        return (
            f"Epoch {epo:04d} | Loss {mean_loss:.6f} | "
            f"MSE {mean_mse:.6f} | LL {mean_ll:.6f}",
            kl_coef,
            {
                "loss": mean_loss,
                "mse": mean_mse,
                "likelihood": mean_ll,
                "kl_first_p": mean_kl,
                "std_first_p": mean_std,
            },
        )


    final_epoch = start_epoch - 1
    last_kl_coef = 1.0
    stopped_early = False

    for epo in range(start_epoch, start_epoch + args.niters):

        message_train, kl_coef, train_stats = train_epoch(epo)
        final_epoch = epo
        last_kl_coef = kl_coef
        stop_training = False

        if epo % n_iters_to_viz == 0:
            if val_encoder is not None:
                model.eval()
                val_res, *_ = compute_loss_all_batches(
                    model,
                    val_encoder,
                    val_graph,
                    val_decoder,
                    n_batches=val_batch,
                    device=device,
                    n_traj_samples=3,
                    kl_coef=kl_coef,
                    collect_outputs=False,
                )

                message_val = 'Epoch {:04d} [Validation] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | KL edge {:.4f} | FP STD {:.4f}|'.format(
                    epo,
                    val_res["loss"], val_res["mse"], val_res["likelihood"],
                    val_res["kl_first_p"], val_res["kl_edge"], val_res["std_first_p"])

                if global_rank == 0:
                    logger.info("Experiment " + str(experimentID))
                    logger.info(message_train)
                    logger.info(message_val)
                    logger.info("KL coef: {}".format(kl_coef))

                    if wandb_run is not None:
                        wandb_run.log({
                            "epoch": epo,
                            "train/loss": train_stats["loss"],
                            "train/mse": train_stats["mse"],
                            "train/likelihood": train_stats["likelihood"],
                            "train/kl_first_p": train_stats["kl_first_p"],
                            "train/std_first_p": train_stats["std_first_p"],
                            "val/loss": float(val_res["loss"]),
                            "val/mse": float(val_res["mse"]),
                            "val/likelihood": float(val_res["likelihood"]),
                            "val/kl_first_p": float(val_res["kl_first_p"]),
                            "val/kl_edge": float(val_res["kl_edge"]),
                            "val/std_first_p": float(val_res["std_first_p"]),
                            "kl_coef": float(kl_coef),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "early_stopping/epochs_without_improvement": epochs_without_improvement,
                        })

                    print("data: %s, encoder: %s, sample: %s, mode:%s" % (
                        args.data_label, args.z0_encoder, str(args.sample_percent_train), args.mode))

                    val_mse = float(val_res["mse"])
                    if val_mse < best_val_mse - args.early_stopping_min_delta:
                        best_val_mse = val_mse
                        best_epoch = epo
                        epochs_without_improvement = 0
                        model_for_state = model.module if hasattr(model, "module") else model
                        best_model_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model_for_state.state_dict().items()
                        }
                        message_best = 'Epoch {:04d} [Validation] | Best mse {:.12f}|'.format(
                            epo, best_val_mse)
                        logger.info(message_best)
                        print(message_best)
                        if wandb_run is not None:
                            wandb_run.summary["best_val_mse"] = best_val_mse
                            wandb_run.summary["best_epoch"] = int(epo)

                        best_checkpoint_path = os.path.join(
                            args.save,
                            f"experiment_{experimentID}_{args.z0_encoder}_{args.data_label}_"
                            f"{args.sample_percent_train}_{args.mode}_epoch_{epo}_val_mse_{best_val_mse}.ckpt"
                        )
                        utils.save_checkpoint_pf(
                            best_checkpoint_path,
                            model_for_state,
                            optimizer,
                            scheduler,
                            epoch=epo,
                            best_val_loss=best_val_mse,
                            args=args,
                            device=device
                        )
                    else:
                        epochs_without_improvement += 1

                    if (
                        args.early_stopping_patience > 0
                        and epochs_without_improvement >= args.early_stopping_patience
                    ):
                        stop_training = True
                        stopped_early = True
                        message_stop = (
                            f"Early stopping at epoch {epo}: validation MSE did not improve "
                            f"by at least {args.early_stopping_min_delta:g} for "
                            f"{args.early_stopping_patience} checks."
                        )
                        logger.info(message_stop)
                        print(message_stop)

        if is_dist:
            stop_tensor = torch.tensor(
                [int(stop_training)], dtype=torch.int32, device=device
            )
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

    if is_dist:
        dist.barrier()

    if global_rank == 0 and test_encoder is not None:
        model_for_eval = model.module if hasattr(model, "module") else model
        if best_model_state is not None:
            model_for_eval.load_state_dict(best_model_state)

        test_res, *_ = compute_loss_all_batches(
            model_for_eval,
            test_encoder,
            test_graph,
            test_decoder,
            n_batches=test_batch,
            device=device,
            n_traj_samples=3,
            kl_coef=last_kl_coef,
            collect_outputs=False,
        )
        message_test = 'Best epoch {:04d} [Test] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | KL edge {:.4f} | FP STD {:.4f}|'.format(
            best_epoch if best_epoch is not None else final_epoch,
            test_res["loss"], test_res["mse"], test_res["likelihood"],
            test_res["kl_first_p"], test_res["kl_edge"], test_res["std_first_p"])
        logger.info(message_test)
        print(message_test)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": final_epoch,
                "test/loss": float(test_res["loss"]),
                "test/mse": float(test_res["mse"]),
                "test/likelihood": float(test_res["likelihood"]),
                "test/kl_first_p": float(test_res["kl_first_p"]),
                "test/kl_edge": float(test_res["kl_edge"]),
                "test/std_first_p": float(test_res["std_first_p"]),
                "test/evidence_mean": float(test_res["evidence_mean"]),
                "test/evidence_std": float(test_res["evidence_std"]),
                "test/q_temporal_variation": float(test_res["q_temporal_variation"]),
                "test/posterior_prior_l1": float(test_res["posterior_prior_l1"]),
                "test/nri_message_norm": float(test_res["nri_message_norm"]),
                "test/latent_derivative_norm": float(test_res["latent_derivative_norm"]),
            })
            wandb_run.summary["best_val_mse"] = float(best_val_mse)
            wandb_run.summary["best_epoch"] = int(best_epoch) if best_epoch is not None else None
            wandb_run.summary["test_mse_at_best_val"] = float(test_res["mse"])
            wandb_run.summary["stopped_early"] = stopped_early
            if best_checkpoint_path is not None:
                wandb_run.summary["best_checkpoint"] = best_checkpoint_path

    if global_rank == 0 and wandb_run is not None:
        wandb_run.finish()

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()

'''
ODE-RNN baseline (Rubanova et al. 2019): one shared, per-node continuous-time RNN, no graph.
Reuses the same corrected dataloaders as run_models_corrected.py (CorrectedParseData /
IEEE39ParseData) for a fair, apples-to-apples comparison against Corrected LG-ODE -- same data,
same splits, same normalization, same observation masks -- but the graph tensor is loaded and
then simply never used (see lib/baseline_odernn.py).
'''
import os
import sys
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from lib.baseline_odernn import ODERNNBaseline
from lib.utils import compute_loss_all_batches

parser = argparse.ArgumentParser('ODE-RNN baseline')
parser.add_argument('--n-balls', type=int, default=5)
parser.add_argument('--niters', type=int, default=30)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--save', type=str, default='experiments_odernn/')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('-r', '--random-seed', type=int, default=1991)
parser.add_argument('--data', type=str, default='spring', help="spring,charged,ieee39")
parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--extrap', type=str, default="False")
parser.add_argument('--sample-percent-train', type=float, default=0.6)
parser.add_argument('--sample-percent-test', type=float, default=0.6)
parser.add_argument('--l2', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default="AdamW")
parser.add_argument('--clip', type=float, default=10)
parser.add_argument('--cutting_edge', type=bool, default=True)
parser.add_argument('--extrap_num', type=int, default=40)
parser.add_argument('--alias', type=str, default="run")
parser.add_argument('--dataset-dir', type=str, default=None)

args = parser.parse_args()

if args.data == "spring":
    args.dataset = 'data/example_data'
    args.suffix = '_springs5'
    args.total_ode_step = 60
elif args.data == "charged":
    args.dataset = 'data/charged'
    args.suffix = '_charged5'
    args.total_ode_step = 60
elif args.data == "ieee39":
    args.dataset = 'data/processed/ieee39_gen'
    args.suffix = '_ieee39gen'
    args.total_ode_step = 60
    args.n_balls = 10

if args.dataset_dir is not None:
    args.dataset = args.dataset_dir

if torch.cuda.is_available():
    print("Using GPU" + "-" * 80)
    device = torch.device("cuda:0")
else:
    print("Using CPU" + "-" * 80)
    device = torch.device("cpu")

if args.extrap == "True":
    print("Running extrap mode" + "-" * 80)
    args.mode = "extrap"
else:
    print("Running interp mode" + "-" * 80)
    args.mode = "interp"


if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    utils.makedirs(args.save)

    experimentID = args.load
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)

    print("Loading dataset: " + args.dataset)
    if args.data == "ieee39":
        from lib.ieee39_dataLoader import IEEE39ParseData
        dataloader = IEEE39ParseData(args.dataset, mode=args.mode, args=args)
    else:
        from lib.corrected_dataLoader import CorrectedParseData
        dataloader = CorrectedParseData(args.dataset, suffix=args.suffix, mode=args.mode, args=args)

    train_encoder, train_decoder, train_graph, train_batch = dataloader.load_data(
        sample_percent=args.sample_percent_train, batch_size=args.batch_size, data_type="train")
    val_encoder, val_decoder, val_graph, val_batch = dataloader.load_data(
        sample_percent=args.sample_percent_test, batch_size=args.batch_size, data_type="val")
    test_encoder, test_decoder, test_graph, test_batch = dataloader.load_data(
        sample_percent=args.sample_percent_test, batch_size=args.batch_size, data_type="test")

    input_dim = dataloader.feature

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    obsrv_std = torch.Tensor([0.01]).to(device)
    model = ODERNNBaseline(input_dim=input_dim, hidden_dim=args.hidden_dim, obsrv_std=obsrv_std, device=device).to(device)

    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)

    log_path = "logs/" + args.alias + "_odernn_" + args.data + "_" + str(args.sample_percent_train) + "_" + args.mode + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)

    best_test_mse = np.inf

    def train_single_batch(batch_dict_encoder, batch_dict_decoder, batch_dict_graph):
        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph)
        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        loss_value = loss.data.item()
        del loss
        torch.cuda.empty_cache()
        return loss_value, train_res["mse"], train_res["likelihood"]

    def train_epoch(epo):
        model.train()
        loss_list, mse_list, likelihood_list = [], [], []
        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):
            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, mse, likelihood = train_single_batch(batch_dict_encoder, batch_dict_decoder, batch_dict_graph)

            loss_list.append(loss), mse_list.append(mse), likelihood_list.append(likelihood)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
            torch.cuda.empty_cache()

        scheduler.step()

        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f}|'.format(
            epo, np.mean(loss_list), np.mean(mse_list), np.mean(likelihood_list))
        return message_train

    for epo in range(1, args.niters + 1):
        message_train = train_epoch(epo)

        model.eval()
        test_res = compute_loss_all_batches(model, test_encoder, test_graph, test_decoder,
                                            n_batches=test_batch, device=device, n_traj_samples=1, kl_coef=0.)

        message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MSE {:.6F} | Likelihood {:.6f}|'.format(
            epo, test_res["loss"], test_res["mse"], test_res["likelihood"])

        logger.info("Experiment " + str(experimentID))
        logger.info(message_train)
        logger.info(message_test)
        print("data: %s, model: ODE-RNN, sample: %s, mode:%s" % (args.data, str(args.sample_percent_train), args.mode))

        if test_res["mse"] < best_test_mse:
            best_test_mse = test_res["mse"]
            message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best mse {:.6f}|'.format(epo, best_test_mse)
            logger.info(message_best)
            ckpt_path = os.path.join(args.save, "experiment_" + str(
                experimentID) + "_odernn_" + args.data + "_" + str(
                args.sample_percent_train) + "_" + args.mode + "_epoch_" + str(epo) + "_mse_" + str(
                best_test_mse) + '.ckpt')
            torch.save({'args': args, 'state_dict': model.state_dict()}, ckpt_path)

        torch.cuda.empty_cache()

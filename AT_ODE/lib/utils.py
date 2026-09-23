import os
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
	torch.save(state, filename)

	
def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger


def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)

def flatten(x, dim):
	return x.reshape(x.size()[:dim] + (-1, ))


def get_device(tensor):
	"""
	Return the device of a torch Tensor (CPU/CUDA/MPS).
	Falls back to CPU for non-tensors / None.
	"""
	if torch.is_tensor(tensor):
		return tensor.device
	return torch.device("cpu")

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

def get_dict_template():
	return {"raw": None,
			"time_setps": None,
			"mask": None
			}
def get_next_batch_new(dataloader,device):
	data_dict = dataloader.__next__()
	#device_now = data_dict.batch.device
	return data_dict.to(device)

def get_next_batch(dataloader,device):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()

	batch_dict = get_dict_template()


	batch_dict["raw"] = data_dict["raw"].to(device)
	batch_dict["time_steps"] = data_dict["time_steps"].to(device)
	batch_dict["mask"] = data_dict["mask"].to(device)

	return batch_dict


def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	checkpt = torch.load(ckpt_path)
	ckpt_args = checkpt['args']
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict) 
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr


def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]

def create_net(n_inputs, n_outputs, n_layers = 1,
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)



def compute_loss_all_batches(model,
                             encoder, graph, decoder,
                             n_batches, device,
                             n_traj_samples=1, kl_coef=1.,
                             collect_outputs=True):
    total = {}
    total["loss"] = 0
    total["likelihood"] = 0
    total["mse"] = 0
    total["kl_first_p"] = 0
    total["std_first_p"] = 0
    total["kl_edge"] = 0
    total["evidence_l1"] = 0
    total["evidence_mean"] = 0
    total["evidence_std"] = 0
    total["q_temporal_variation"] = 0
    total["posterior_prior_l1"] = 0
    total["nri_message_norm"] = 0
    total["latent_derivative_norm"] = 0
    all_pred_samples = []   # <-- NEW
    all_dec_batches  = []   # <-- NEW
    n_test_batches = 0

    model.eval()
    model_for_loss = model.module if hasattr(model, "module") else model
    print("Computing loss... ")
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            batch_dict_encoder = get_next_batch_new(encoder, device)
            batch_dict_graph = get_next_batch_new(graph, device)
            batch_dict_decoder = get_next_batch(decoder, device)

            results = model_for_loss.compute_all_losses(
                batch_dict_encoder,
                batch_dict_decoder,
                batch_dict_graph,
                n_traj_samples=n_traj_samples,
                kl_coef=kl_coef,
            )

            for key in total.keys():
                if key in results:
                    var = results[key]
                    if isinstance(var, torch.Tensor):
                        var = var.detach().item()
                    total[key] += var


            if collect_outputs:
                all_pred_samples.append(results["pred_samples"].detach())  # [S,M,T,D]
                kept_dec = {
                    "time_steps": batch_dict_decoder["time_steps"].detach().clone(),
                    "raw": batch_dict_decoder["raw"].detach().clone(),
                    "mask": batch_dict_decoder["mask"].detach().clone(),
                    "types": batch_dict_decoder.get("types", None),
                }
                if kept_dec["types"] is not None:
                    kept_dec["types"] = kept_dec["types"].detach().clone()
                all_dec_batches.append(kept_dec)

            n_test_batches += 1

            # del batch_dict_encoder, batch_dict_graph, batch_dict_decoder, results

        if n_test_batches > 0:
            for key, value in total.items():
                total[key] = total[key] / n_test_batches

    return total, batch_dict_encoder, batch_dict_graph, batch_dict_decoder, {
        "pred_samples": all_pred_samples,   # list of [S,M,T,D]
        "dec_batches":  all_dec_batches,    # list of dicts
    }

def set_global_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

def get_global_args():
    return _GLOBAL_ARGS


def save_checkpoint_pf(path, model, optimizer, scheduler, epoch, best_val_loss, args, device):
    """
    Minimal, safe checkpoint for your PF script.
    Saves: epoch, best_val_loss, args, model, optimizer, scheduler.
    Atomic write via tmp -> replace.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "args": vars(args),  # store CLI args as plain dict (pickleable)
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        }
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def resume_from_pf(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
    strict=True,
    return_checkpoint=False,
):
    """
    Load PF checkpoint saved by save_checkpoint_pf().
    Returns: start_epoch, best_val_loss
    """
    # PF checkpoints contain optimizer, scheduler, NumPy state, and argument
    # objects, so trusted local checkpoints require the full PyTorch loader.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    state_dict = ckpt["model"]
    ckpt_is_ddp = any(key.startswith("module.") for key in state_dict)
    model_is_ddp = hasattr(model, "module")
    if ckpt_is_ddp and not model_is_ddp:
        state_dict = {
            key[len("module."):]: value for key, value in state_dict.items()
        }
    elif model_is_ddp and not ckpt_is_ddp:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    rng = ckpt.get("rng", None)
    if rng is not None:
        tstate = rng.get("torch", None)
        if tstate is not None:
            if isinstance(tstate, torch.Tensor):
                tstate = tstate.to(dtype=torch.uint8, device="cpu")
            torch.set_rng_state(tstate)

        if torch.cuda.is_available() and rng.get("cuda") is not None:
            for i, state in enumerate(rng["cuda"]):
                if state is not None and i < torch.cuda.device_count():
                    if isinstance(state, torch.Tensor):
                        state = state.to(dtype=torch.uint8, device="cpu")
                    else:
                        state = torch.tensor(state, dtype=torch.uint8)
                    torch.cuda.set_rng_state(state, device=i)

        np.random.set_state(rng["numpy"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    if return_checkpoint:
        return start_epoch, best_val_loss, ckpt
    return start_epoch, best_val_loss


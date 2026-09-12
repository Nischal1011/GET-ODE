'''
Build the IEEE39-Gen forecasting dataset from the raw Mendeley transient-stability-assessment
pickle (data/raw/ieee39_tsa/tsa_data.pkl).

Treats each of the 10 generators as a node with a COMPLETE interaction-support graph (every
pair can message-pass). This is an interaction-support assumption, not the physical IEEE-39
transmission-line topology -- no ANDES, no admittance data, no physical graph is used or
derived here. See reports/DATA_CHARACTERISTICS.md for why the physical topology isn't
available in this repo.

Usage:
    python data/prepare_ieee39_gen.py
'''
import os
import pickle
import numpy as np
import pandas as pd

RAW_PATH = 'data/raw/ieee39_tsa/tsa_data.pkl'
OUT_DIR = 'data/processed/ieee39_gen'
OUT_PATH = os.path.join(OUT_DIR, 'ieee39_gen.npz')

NUM_GENERATORS = 10
NUM_TIMES = 60
CONTEXT_END = 30  # times [0, CONTEXT_END) are context, [CONTEXT_END, NUM_TIMES) are forecast
RATIOS = [0.4, 0.6, 0.8]
SEED = 1991

GENERATOR_IDS = np.array([f"G{str(i).zfill(2)}" for i in range(1, NUM_GENERATORS + 1)])
FEATURE_NAMES = np.array(["P", "ut", "ie", "xspeed", "firel"])


def load_raw():
    with open(RAW_PATH, 'rb') as f:
        data, target = pickle.load(f)
    return data, target


def convert_trajectories(data):
    '''
    Each simulation is a [60, 50] DataFrame, feature-major columns:
      10 P | 10 ut | 10 ie | 10 xspeed | 10 firel
    values.reshape(60, 5, 10).transpose(0, 2, 1) -> [60 times, 10 generators, 5 features],
    with feature order [P, ut, ie, xspeed, firel] preserved.
    '''
    n = len(data)
    states = np.empty((n, NUM_TIMES, NUM_GENERATORS, len(FEATURE_NAMES)), dtype=np.float32)
    times = np.empty((n, NUM_TIMES), dtype=np.float32)

    for i, df in enumerate(data):
        values = df.to_numpy(dtype=np.float32)  # [60, 50]
        states[i] = values.reshape(NUM_TIMES, 5, NUM_GENERATORS).transpose(0, 2, 1)
        original_times = df.index.to_numpy(dtype=np.float32)
        times[i] = original_times - original_times[0]

    return states, times


def build_complete_graph():
    adjacency = np.ones((NUM_GENERATORS, NUM_GENERATORS), dtype=np.float32)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def clearing_time_groups(times):
    '''Recover the discrete clearing-time group from each trajectory's ORIGINAL absolute start
    time (before rebasing). Caller passes the raw per-trajectory start offsets.'''
    unique_starts, group_ids = np.unique(np.round(times, 3), return_inverse=True)
    return group_ids, unique_starts


def stratified_split(labels, groups, seed):
    '''80/10/10 split by whole trajectory, stratified by (label, clearing-time group).'''
    rng = np.random.RandomState(seed)
    n = len(labels)
    strata = labels.astype(np.int64) * 100 + groups.astype(np.int64)

    train_idx, val_idx, test_idx = [], [], []
    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        rng.shuffle(idx)
        n_s = len(idx)
        n_train = int(round(0.8 * n_s))
        n_val = int(round(0.1 * n_s))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    test_idx = np.sort(np.concatenate(test_idx))
    assert len(train_idx) + len(val_idx) + len(test_idx) == n
    return train_idx, val_idx, test_idx


def compute_normalization(states, train_idx):
    train_states = states[train_idx]
    mean = train_states.mean(axis=(0, 1, 2), keepdims=True)
    std = train_states.std(axis=(0, 1, 2), keepdims=True)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def make_interp_mask(n, ratio, rng):
    n_obs = round(ratio * NUM_TIMES)
    mask = np.zeros((n, NUM_TIMES, NUM_GENERATORS), dtype=np.uint8)
    for i in range(n):
        for g in range(NUM_GENERATORS):
            obs_idx = rng.choice(NUM_TIMES, size=n_obs, replace=False)
            mask[i, obs_idx, g] = 1
    return mask


def make_extrap_mask(n, ratio, rng):
    n_obs = round(ratio * CONTEXT_END)
    mask = np.zeros((n, NUM_TIMES, NUM_GENERATORS), dtype=np.uint8)
    for i in range(n):
        for g in range(NUM_GENERATORS):
            obs_idx = rng.choice(CONTEXT_END, size=n_obs, replace=False)
            mask[i, obs_idx, g] = 1
    return mask


def run_checks(states, times, adjacency, labels, train_idx, val_idx, test_idx,
                interp_masks, extrap_masks):
    assert states.shape == (len(labels), NUM_TIMES, NUM_GENERATORS, len(FEATURE_NAMES)), states.shape
    assert times.shape == (len(labels), NUM_TIMES), times.shape
    assert adjacency.shape == (NUM_GENERATORS, NUM_GENERATORS)
    assert np.isfinite(states).all(), "non-finite values in states"
    assert np.isfinite(times).all(), "non-finite values in times"
    assert np.all(np.diff(times, axis=1) > 0), "times are not strictly increasing"
    assert np.all(np.diag(adjacency) == 0), "graph diagonal is not zero"
    assert np.all(adjacency[~np.eye(NUM_GENERATORS, dtype=bool)] == 1), "off-diagonal not all 1"

    train_set, val_set, test_set = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    assert not (train_set & val_set), "train/val overlap"
    assert not (train_set & test_set), "train/test overlap"
    assert not (val_set & test_set), "val/test overlap"
    assert len(train_set | val_set | test_set) == len(labels), "split doesn't cover all trajectories"

    for name, mask in interp_masks.items():
        assert mask.shape == (len(labels), NUM_TIMES, NUM_GENERATORS), (name, mask.shape)
        assert (mask.sum(axis=1) > 0).all(), f"{name}: some generator has zero observations"

    for name, mask in extrap_masks.items():
        assert mask.shape == (len(labels), NUM_TIMES, NUM_GENERATORS), (name, mask.shape)
        assert (mask[:, CONTEXT_END:, :] == 0).all(), f"{name}: observed points after context boundary"
        assert (mask[:, :CONTEXT_END, :].sum(axis=1) > 0).all(), f"{name}: some generator has zero context observations"

    print("All checks passed.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading raw pickle...")
    data, target = load_raw()
    labels = target.to_numpy(dtype=np.int64)

    print("Converting trajectories (feature-major [60,50] -> [60,10,5])...")
    states, times = convert_trajectories(data)

    print("Building complete interaction-support graph...")
    adjacency = build_complete_graph()

    print("Recovering clearing-time groups from original absolute start times...")
    original_starts = np.array([df.index.to_numpy(dtype=np.float32)[0] for df in data])
    groups, unique_starts = clearing_time_groups(original_starts)
    print(f"  {len(unique_starts)} distinct clearing-time groups: {unique_starts}")

    print("Building stratified 80/10/10 split (seed=%d)..." % SEED)
    train_idx, val_idx, test_idx = stratified_split(labels, groups, SEED)
    print(f"  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    print("Computing normalization statistics (train only)...")
    mean, std = compute_normalization(states, train_idx)

    print("Generating interpolation masks...")
    rng = np.random.RandomState(SEED)
    interp_masks = {f"interp_mask_{int(r*100)}": make_interp_mask(len(labels), r, rng) for r in RATIOS}

    print("Generating extrapolation masks...")
    extrap_masks = {f"extrap_mask_{int(r*100)}": make_extrap_mask(len(labels), r, rng) for r in RATIOS}

    print("Running checks...")
    run_checks(states, times, adjacency, labels, train_idx, val_idx, test_idx,
               interp_masks, extrap_masks)

    print(f"Saving to {OUT_PATH} ...")
    np.savez_compressed(
        OUT_PATH,
        states=states.astype(np.float32),
        times=times.astype(np.float32),
        adjacency=adjacency,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_mean=mean,
        feature_std=std,
        interp_mask_40=interp_masks["interp_mask_40"],
        interp_mask_60=interp_masks["interp_mask_60"],
        interp_mask_80=interp_masks["interp_mask_80"],
        extrap_mask_40=extrap_masks["extrap_mask_40"],
        extrap_mask_60=extrap_masks["extrap_mask_60"],
        extrap_mask_80=extrap_masks["extrap_mask_80"],
        generator_ids=GENERATOR_IDS,
        feature_names=FEATURE_NAMES,
        clearing_time_group=groups,
    )
    print("Done.")


if __name__ == '__main__':
    main()

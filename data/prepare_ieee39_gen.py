'''
Build the IEEE39-Gen forecasting dataset from the raw Mendeley transient-stability-assessment
pickle (data/raw/ieee39_tsa/tsa_data.pkl).

Graph: a physically-derived generator-to-generator interaction-support graph, Kron-reduced from
the real IEEE 39-bus network's admittance matrix (data/build_ieee39_kron_graph.py, MATPOWER
case39.m bus/branch parameters -- see that script's docstring for the full method and citation
chain), NOT a complete-graph placeholder. Run data/build_ieee39_kron_graph.py first to produce
data/ieee39_kron_reduced.npz, which this script loads.

Usage:
    python data/build_ieee39_kron_graph.py          # once, produces the graph
    python data/prepare_ieee39_gen.py                                    # full-scale (~80/10/10)
    python data/prepare_ieee39_gen.py --train-n 7000 --val-n 1000 --test-n 2000 --out-dir data/processed/ieee39_gen_subset
'''
import argparse
import os
import pickle
import numpy as np
import pandas as pd

RAW_PATH = 'data/raw/ieee39_tsa/tsa_data.pkl'
KRON_PATH = 'data/ieee39_kron_reduced.npz'
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


def build_kron_reduced_graph():
    d = np.load(KRON_PATH)
    adjacency = d['binary_adjacency'].astype(np.float32)
    assert adjacency.shape == (NUM_GENERATORS, NUM_GENERATORS)
    return adjacency


def clearing_time_groups(times):
    '''Recover the discrete clearing-time group from each trajectory's ORIGINAL absolute start
    time (before rebasing). Caller passes the raw per-trajectory start offsets.'''
    unique_starts, group_ids = np.unique(np.round(times, 3), return_inverse=True)
    return group_ids, unique_starts


def stratified_split(labels, groups, seed, target_counts=None):
    '''
    Split by whole trajectory, stratified by (label, clearing-time group). Default: 80/10/10 of
    the full corpus. If target_counts=(n_train, n_val, n_test) is given (a fixed-size subset,
    e.g. for the TMLR-scale comparison), each stratum contributes proportionally to its share of
    the full population, via the same largest-remainder rounding used for the springs/charged
    subsets (data/make_subset.py), so the subset's (label, clearing-time) distribution matches
    the full corpus as closely as an integer sample allows.
    '''
    rng = np.random.RandomState(seed)
    n = len(labels)
    strata = labels.astype(np.int64) * 100 + groups.astype(np.int64)
    unique_strata = np.unique(strata)

    if target_counts is None:
        train_idx, val_idx, test_idx = [], [], []
        for stratum in unique_strata:
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

    n_train_target, n_val_target, n_test_target = target_counts
    out = {'train': [], 'val': [], 'test': []}
    for split_name, target_n in [('train', n_train_target), ('val', n_val_target), ('test', n_test_target)]:
        raw_quota = {s: (strata == s).sum() / n * target_n for s in unique_strata}
        quota = {s: int(np.floor(q)) for s, q in raw_quota.items()}
        remainder = target_n - sum(quota.values())
        fracs = sorted(raw_quota.items(), key=lambda kv: kv[1] - quota[kv[0]], reverse=True)
        for s, _ in fracs[:remainder]:
            quota[s] += 1
        out[split_name] = quota

    train_idx, val_idx, test_idx = [], [], []
    for stratum in unique_strata:
        idx = np.where(strata == stratum)[0].tolist()
        rng.shuffle(idx)
        n_train = min(out['train'][stratum], len(idx))
        take_train, idx = idx[:n_train], idx[n_train:]
        n_val = min(out['val'][stratum], len(idx))
        take_val, idx = idx[:n_val], idx[n_val:]
        n_test = min(out['test'][stratum], len(idx))
        take_test = idx[:n_test]
        train_idx.extend(take_train)
        val_idx.extend(take_val)
        test_idx.extend(take_test)

    train_idx = np.sort(np.array(train_idx, dtype=np.int64))
    val_idx = np.sort(np.array(val_idx, dtype=np.int64))
    test_idx = np.sort(np.array(test_idx, dtype=np.int64))
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
    assert len(set(val_idx.tolist()) & set(test_idx.tolist())) == 0
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
    assert np.all((adjacency == 0) | (adjacency == 1)), "adjacency must be binary"
    assert set(np.unique(adjacency).tolist()) == {0.0, 1.0}, "Kron-reduced graph should have both edges and non-edges"

    train_set, val_set, test_set = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    assert not (train_set & val_set), "train/val overlap"
    assert not (train_set & test_set), "train/test overlap"
    assert not (val_set & test_set), "val/test overlap"
    assert len(train_set | val_set | test_set) <= len(labels), "split indices out of range"

    for name, mask in interp_masks.items():
        assert mask.shape == (len(labels), NUM_TIMES, NUM_GENERATORS), (name, mask.shape)
        assert (mask.sum(axis=1) > 0).all(), f"{name}: some generator has zero observations"

    for name, mask in extrap_masks.items():
        assert mask.shape == (len(labels), NUM_TIMES, NUM_GENERATORS), (name, mask.shape)
        assert (mask[:, CONTEXT_END:, :] == 0).all(), f"{name}: observed points after context boundary"
        assert (mask[:, :CONTEXT_END, :].sum(axis=1) > 0).all(), f"{name}: some generator has zero context observations"

    print("All checks passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-n', type=int, default=None)
    parser.add_argument('--val-n', type=int, default=None)
    parser.add_argument('--test-n', type=int, default=None)
    parser.add_argument('--out-dir', type=str, default=OUT_DIR)
    args = parser.parse_args()
    target_counts = None
    if args.train_n is not None:
        assert args.val_n is not None and args.test_n is not None, "give all of --train-n/--val-n/--test-n together"
        target_counts = (args.train_n, args.val_n, args.test_n)

    out_dir = args.out_dir
    out_path = os.path.join(out_dir, 'ieee39_gen.npz')
    os.makedirs(out_dir, exist_ok=True)

    print("Loading raw pickle...")
    data, target = load_raw()
    labels = target.to_numpy(dtype=np.int64)

    print("Converting trajectories (feature-major [60,50] -> [60,10,5])...")
    states, times = convert_trajectories(data)

    print("Loading Kron-reduced interaction-support graph...")
    adjacency = build_kron_reduced_graph()

    print("Recovering clearing-time groups from original absolute start times...")
    original_starts = np.array([df.index.to_numpy(dtype=np.float32)[0] for df in data])
    groups, unique_starts = clearing_time_groups(original_starts)
    print(f"  {len(unique_starts)} distinct clearing-time groups: {unique_starts}")

    if target_counts is None:
        print("Building stratified 80/10/10 split (seed=%d)..." % SEED)
    else:
        print(f"Building fixed-size stratified split (train={target_counts[0]}, "
              f"val={target_counts[1]}, test={target_counts[2]}, seed={SEED})...")
    train_idx, val_idx, test_idx = stratified_split(labels, groups, SEED, target_counts=target_counts)
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

    print(f"Saving to {out_path} ...")
    np.savez_compressed(
        out_path,
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

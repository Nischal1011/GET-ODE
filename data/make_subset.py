'''
Builds a fixed, deterministic, stratified subset of the full springs/charged datasets for the
TMLR-bound comparison: a much smaller sample size than the paper's 20k/5k, chosen not because
sample count is the scientific contribution, but because the defining behavior (interaction
structure, irregular per-node observations, interp/extrap tasks) is preserved by a properly
stratified subset, and the full 20k/18k-trajectory runs are far too slow to iterate on during
architecture development.

Stratification key:
- springs: number of active (nonzero) edges per trajectory (0-10 for 5 particles), preserving
  the sparse-graph density distribution -- an edge means a real Hooke's-law interaction, absence
  means none, so the density distribution is the behavior that must be preserved.
- charged: number of positive (attractive) edges per trajectory (4-10; every off-diagonal pair
  is always connected, only the sign varies), preserving the attraction/repulsion balance --
  this dataset is dense, not sparse, so density isn't the relevant axis, sign balance is.

Selection is stratified-proportional sampling without replacement, from a fixed-seed RNG, over
the full train pool (for train+val) and the full test set (for test) independently. The train
pool is shuffled (same fixed seed) after selection so a downstream contiguous "last N% is val"
split (CorrectedParseData's own mechanism) isn't biased toward any stratification bucket.

Usage:
    python data/make_subset.py --dataset spring  --train-pool-size 6000 --test-size 1000 --seed 20260916
    python data/make_subset.py --dataset charged --train-pool-size 6000 --test-size 1000 --seed 20260916

Output: data/<dataset>_subset/{loc,vel,edges,times}_{train,test}_<suffix>.npy (same naming
convention as data/spring, data/charged, so existing dataloaders work unmodified via
--dataset-dir data/<dataset>_subset), plus subset_manifest.json recording the exact selected
original indices, a sha256 of every saved file, and the full-vs-subset stratification-key
distribution for both the train pool and the test set.
'''
import argparse
import hashlib
import json
import os

import numpy as np


def edge_count_key(edges):
    return int(np.sum(np.triu(edges, k=1) != 0))


def pos_edge_count_key(edges):
    return int(np.sum(np.triu(edges, k=1) > 0))


KEY_FN = {'spring': edge_count_key, 'charged': pos_edge_count_key}
SUFFIX = {'spring': '_springs5', 'charged': '_charged5'}
SRC_DIR = {'spring': 'data/spring', 'charged': 'data/charged'}


def stratified_sample(keys, target_n, rng):
    '''Proportional-to-population stratified sample of target_n indices out of len(keys), no
    replacement, using an explicit per-bucket quota (largest-remainder rounding) so the subset's
    bucket proportions match the full population's as closely as an integer sample allows.'''
    keys = np.asarray(keys)
    n = len(keys)
    buckets = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)

    raw_quota = {k: len(idxs) / n * target_n for k, idxs in buckets.items()}
    quota = {k: int(np.floor(q)) for k, q in raw_quota.items()}
    remainder = target_n - sum(quota.values())
    # Largest-remainder method: give the leftover slots to the buckets with the biggest
    # fractional part, so rounding doesn't systematically favor small or large buckets.
    fracs = sorted(raw_quota.items(), key=lambda kv: kv[1] - quota[kv[0]], reverse=True)
    for k, _ in fracs[:remainder]:
        quota[k] += 1

    selected = []
    for k, idxs in buckets.items():
        q = min(quota.get(k, 0), len(idxs))
        selected.extend(rng.choice(idxs, size=q, replace=False).tolist())
    # Top up if integer rounding left us short (bucket ran out of members) -- fill from whatever
    # remains, still without replacement.
    if len(selected) < target_n:
        remaining = [i for i in range(n) if i not in set(selected)]
        extra = rng.choice(remaining, size=target_n - len(selected), replace=False).tolist()
        selected.extend(extra)
    rng.shuffle(selected)
    return selected[:target_n]


def distribution(keys, indices):
    sub = np.asarray(keys)[indices]
    vals, counts = np.unique(sub, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['spring', 'charged'], required=True)
    p.add_argument('--train-pool-size', type=int, default=6000)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20260916)
    p.add_argument('--out-dir', type=str, default=None)
    args = p.parse_args()

    src = SRC_DIR[args.dataset]
    suffix = SUFFIX[args.dataset]
    key_fn = KEY_FN[args.dataset]
    out_dir = args.out_dir or f'data/{args.dataset}_subset'
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    manifest = {'dataset': args.dataset, 'seed': args.seed, 'source_dir': src,
                'train_pool_size': args.train_pool_size, 'test_size': args.test_size}

    for split, target_n in [('train', args.train_pool_size), ('test', args.test_size)]:
        loc = np.load(f'{src}/loc_{split}{suffix}.npy', allow_pickle=True)
        vel = np.load(f'{src}/vel_{split}{suffix}.npy', allow_pickle=True)
        edges = np.load(f'{src}/edges_{split}{suffix}.npy', allow_pickle=True)
        times = np.load(f'{src}/times_{split}{suffix}.npy', allow_pickle=True)

        keys = [key_fn(edges[i]) for i in range(len(edges))]
        selected = stratified_sample(keys, target_n, rng)

        full_dist = distribution(keys, list(range(len(keys))))
        sub_dist = distribution(keys, selected)
        full_frac = {k: v / len(keys) for k, v in full_dist.items()}
        sub_frac = {k: v / len(selected) for k, v in sub_dist.items()}

        np.save(f'{out_dir}/loc_{split}{suffix}.npy', loc[selected], allow_pickle=True)
        np.save(f'{out_dir}/vel_{split}{suffix}.npy', vel[selected], allow_pickle=True)
        np.save(f'{out_dir}/edges_{split}{suffix}.npy', edges[selected])
        np.save(f'{out_dir}/times_{split}{suffix}.npy', times[selected], allow_pickle=True)

        manifest[split] = {
            'selected_original_indices': [int(i) for i in selected],
            'full_population_size': len(keys),
            'full_key_distribution_frac': full_frac,
            'subset_key_distribution_frac': sub_frac,
            'file_sha256': {
                name: sha256_of(f'{out_dir}/{name}_{split}{suffix}.npy')
                for name in ['loc', 'vel', 'edges', 'times']
            },
        }

        print(f'--- {args.dataset} {split}: {len(selected)}/{len(keys)} selected ---')
        for k in sorted(full_frac):
            print(f'  key={k:>2}  full={full_frac[k]:.4f}  subset={sub_frac.get(k, 0.0):.4f}')

    manifest_path = f'{out_dir}/subset_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nWrote {out_dir}/ (loc/vel/edges/times x train/test) + {manifest_path}')


if __name__ == '__main__':
    main()

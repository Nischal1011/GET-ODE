"""Model-independent deterministic observation-mask selection."""

from __future__ import annotations

import hashlib

import numpy as np


MASK_ALGORITHM_VERSION = "csg-keyed-blake2b-v1"


def deterministic_observation_indices(
    context_length: int,
    observation_ratio: float,
    *,
    global_seed: int,
    split: str,
    sample_index: int,
    node_index: int,
) -> np.ndarray:
    """Select LG-ODE second-stage indices independent of access or shuffle order."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if not 0.0 < observation_ratio <= 1.0:
        raise ValueError("observation_ratio must be in (0, 1]")
    payload = (
        f"{MASK_ALGORITHM_VERSION}|{global_seed}|{split}|{sample_index}|"
        f"{node_index}|{observation_ratio:.8f}"
    ).encode("ascii")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    keep_count = max(1, int(context_length * observation_ratio))
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(context_length, size=keep_count, replace=False))

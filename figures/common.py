'''Loading helpers for the JSONL files written by eval_gilode.py / eval_baseline.py.'''
import json
from collections import defaultdict

import numpy as np


def read_jsonl(*paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def group(rows, keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return xs.mean(0), (xs.std(0, ddof=1) if len(xs) > 1 else np.zeros_like(xs.mean(0)))

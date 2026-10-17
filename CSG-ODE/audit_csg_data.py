#!/usr/bin/env python3
"""Audit all four CSG-ODE data adapters without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.baselines.csg_ode.communicability import CommunicabilityCache
from lib.baselines.csg_ode.config import paper_config
from lib.baselines.csg_ode.data_adapter import DATASETS, build_datasets


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR.parent / "data")
    parser.add_argument(
        "--output", type=Path, default=SCRIPT_DIR / "reports" / "dataset_audit.json"
    )
    parser.add_argument("--seed", type=int, default=1991)
    args = parser.parse_args()

    cache = CommunicabilityCache()
    report = {}
    for dataset in DATASETS:
        config = paper_config(dataset, seed=args.seed)
        bundle = build_datasets(config, args.data_root, communicability_cache=cache)
        sample = bundle.train[0]
        bundle.audit["observation_counts"] = {
            "train": bundle.train.observation_count_summary(),
            "validation": bundle.validation.observation_count_summary(),
            "test": bundle.test.observation_count_summary(),
        }
        bundle.audit["communicability_validation"] = cache.validate(sample["original_adjacency"])
        report[dataset] = bundle.audit
        print(
            f"{dataset}: train={bundle.audit['train_source']['sequences']} "
            f"test={bundle.audit['test_source']['sequences']} "
            f"nodes={bundle.audit['train_source']['nodes']} "
            f"features={bundle.audit['train_source']['features']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from conftest import DATA_ROOT, REPOSITORY_ROOT
from lib.baselines.csg_ode.config import paper_config
from lib.baselines.csg_ode.data_adapter import (
    CSGODEDataset,
    audit_pems_feature_semantics,
    build_datasets,
    collate_csg_batches,
)
from lib.baselines.csg_ode.masking import deterministic_observation_indices


EXPECTED = {
    "springs": (20000, 5000, 5, 4),
    "charged": (20000, 5000, 5, 4),
    "motion_walk": (16, 7, 29, 6),
    "pems08": (199, 49, 170, 3),
}


def test_public_mask_helper_is_order_independent() -> None:
    arguments = {
        "context_length": 40,
        "observation_ratio": 0.6,
        "global_seed": 1991,
        "split": "train",
        "sample_index": 17,
        "node_index": 3,
    }
    first = deterministic_observation_indices(**arguments)
    _ = deterministic_observation_indices(**{**arguments, "sample_index": 9})
    repeated = deterministic_observation_indices(**arguments)
    assert first.tolist() == repeated.tolist()
    assert len(first) == int(40 * 0.6)


@pytest.mark.parametrize("dataset", EXPECTED)
def test_paper_counts_and_standard_batch(dataset: str) -> None:
    config = paper_config(dataset, obs_ratio=0.4)
    bundle = build_datasets(config, DATA_ROOT)
    expected_train, expected_test, nodes, features = EXPECTED[dataset]
    assert bundle.audit["train_source"]["sequences"] == expected_train
    assert bundle.audit["test_source"]["sequences"] == expected_test
    assert bundle.audit["train_source"]["nodes"] == nodes
    assert bundle.audit["train_source"]["features"] == features

    sample = bundle.train[0]
    batch = collate_csg_batches([sample])
    required = {
        "full_values",
        "observed_values",
        "observed_mask",
        "feature_mask",
        "encoder_times",
        "target_times",
        "target_values",
        "target_mask",
        "original_adjacency",
        "edge_type",
        "time_valid_mask",
    }
    assert required <= set(batch)
    assert batch["observed_values"].shape[2:] == (nodes, features)
    assert batch["target_values"].shape[2:] == (nodes, features)
    assert batch["encoder_times"].min() >= 0
    assert batch["encoder_times"].max() <= 1
    assert batch["target_times"].min() >= 0
    assert batch["target_times"].max() <= 1


@pytest.mark.parametrize("dataset", EXPECTED)
def test_masks_are_deterministic_independent_and_ratio_controlled(dataset: str) -> None:
    config = paper_config(dataset, obs_ratio=0.4)
    bundle = build_datasets(config, DATA_ROOT)
    first = bundle.train[0]
    repeated = bundle.train[0]
    torch.testing.assert_close(first["observed_mask"], repeated["observed_mask"])

    node_times = [
        set(first["encoder_times_raw"][first["observed_mask"][:, node]].tolist())
        for node in range(config.num_nodes)
    ]
    assert len({tuple(sorted(values)) for values in node_times}) > 1

    counts_by_ratio = []
    for ratio in (0.4, 0.6, 0.8):
        ratio_config = replace(config, obs_ratio=ratio)
        view = CSGODEDataset(
            bundle.train.storage,
            bundle.train.indices[:1],
            ratio_config,
            logical_split="train",
            normalization_scale=bundle.train.normalization_scale,
        )
        ratio_sample = view[0]
        expected_counts = []
        source_index = int(view.indices[0])
        for node in range(config.num_nodes):
            context_times, _, _, _ = view._node_segments(source_index, node)
            expected_counts.append(max(1, int(len(context_times) * ratio)))
        assert ratio_sample["observation_counts"].tolist() == expected_counts
        counts_by_ratio.append(ratio_sample["observation_counts"])
    assert torch.all(counts_by_ratio[0] <= counts_by_ratio[1])
    assert torch.all(counts_by_ratio[1] <= counts_by_ratio[2])


@pytest.mark.parametrize("dataset", EXPECTED)
def test_extrapolation_has_no_future_encoder_leakage(dataset: str) -> None:
    config = paper_config(dataset, task="extrap", obs_ratio=0.6)
    bundle = build_datasets(config, DATA_ROOT)
    sample = bundle.test[0]
    definition = bundle.test.storage.definition
    assert sample["encoder_times_raw"].max() < definition.context_steps
    assert sample["target_times_raw"].min() >= definition.context_steps
    assert not sample["target_observed_mask"].any()


def test_verified_relation_and_self_edge_conventions() -> None:
    springs = build_datasets(paper_config("springs"), DATA_ROOT).train[0]
    assert not springs["original_adjacency"].diagonal().any()
    assert set(springs["edge_type"].unique().tolist()) == {-1, 0, 1}

    charged = build_datasets(paper_config("charged"), DATA_ROOT).train[0]
    assert charged["original_adjacency"].sum() == 25
    assert charged["original_adjacency"].diagonal().all()
    assert set(charged["edge_type"].unique().tolist()) == {0, 1}

    motion = build_datasets(paper_config("motion_walk"), DATA_ROOT).train[0]
    assert motion["original_adjacency"].sum() == 56
    pems = build_datasets(paper_config("pems08"), DATA_ROOT).train[0]
    assert pems["original_adjacency"].sum() == 548


def test_pems_preprocessor_uses_non_overlapping_global_slices() -> None:
    source = (
        REPOSITORY_ROOT / "data/real/pems08/raw/preprocess_pems08_lg_ode_at_ode.py"
    ).read_text(encoding="utf-8")
    assert "train_raw = raw[:args.train_steps]" in source
    assert "test_raw = raw[args.train_steps:args.train_steps + args.test_steps]" in source


def test_local_pems_feature_semantics_are_never_silently_assumed() -> None:
    bundle = build_datasets(paper_config("pems08"), DATA_ROOT)
    semantics = bundle.audit["feature_semantics"]
    if bundle.audit["paper_compatible"]:
        assert semantics["status"] == "verified_by_sidecar"
        assert semantics["declared_features"] == ["traffic_flow", "speed", "occupancy"]
        assert semantics["declared_raw_sha256"] == semantics["raw_sha256"]
    else:
        assert semantics["status"] in {"mismatch", "unverified", "invalid_sidecar"}
        assert semantics["reason"]


def test_pems_calendar_covariate_variant_is_detected(tmp_path) -> None:
    raw_directory = tmp_path / "real/pems08/raw"
    raw_directory.mkdir(parents=True)
    time_index = np.arange(16)
    raw = np.zeros((16, 170, 3), dtype=np.float64)
    raw[..., 0] = np.arange(170)[None, :]
    raw[..., 1] = ((time_index % 288) / 288.0)[:, None]
    raw[..., 2] = ((time_index // 288) % 7)[:, None]
    np.savez(raw_directory / "data.npz", data=raw)

    semantics = audit_pems_feature_semantics(tmp_path)
    assert not semantics["paper_compatible"]
    assert semantics["detected_features"] == [
        "traffic_flow",
        "time_of_day",
        "day_of_week",
    ]


def test_pems_feature_semantics_sidecar_is_bound_to_raw_checksum(tmp_path) -> None:
    raw_directory = tmp_path / "real/pems08/raw"
    raw_directory.mkdir(parents=True)
    raw_path = raw_directory / "data.npz"
    rng = np.random.default_rng(1991)
    np.savez(raw_path, data=rng.normal(size=(8, 170, 3)).astype(np.float32))
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    sidecar_path = raw_directory / "feature_semantics.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "features": ["traffic_flow", "speed", "occupancy"],
                "raw_sha256": digest,
                "source": "unit-test fixture",
            }
        ),
        encoding="utf-8",
    )

    verified = audit_pems_feature_semantics(tmp_path)
    assert verified["paper_compatible"]
    assert verified["status"] == "verified_by_sidecar"

    sidecar_path.write_text(
        json.dumps(
            {
                "features": ["traffic_flow", "speed", "occupancy"],
                "raw_sha256": "stale-checksum",
            }
        ),
        encoding="utf-8",
    )
    stale = audit_pems_feature_semantics(tmp_path)
    assert not stale["paper_compatible"]
    assert "checksum" in stale["reason"]

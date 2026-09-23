"""Paper-aware adapters from the repository's irregular NumPy datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .communicability import CommunicabilityCache, DEFAULT_COMMUNICABILITY_CACHE
from .config import CSGODEConfig, canonical_dataset_name
from .masking import MASK_ALGORITHM_VERSION, deterministic_observation_indices


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    relative_directory: str
    suffix: str
    use_velocity: bool
    context_steps: int
    future_steps: int
    expected_train: int
    expected_test: int
    expected_nodes: int
    expected_features: int
    expected_edges: int | None


DATASETS: dict[str, DatasetDefinition] = {
    "springs": DatasetDefinition(
        "springs", "synthetic/spring", "springs5", True, 60, 60, 20000, 5000, 5, 4, None
    ),
    "charged": DatasetDefinition(
        "charged", "synthetic/charged", "charged5", True, 60, 60, 20000, 5000, 5, 4, 25
    ),
    "motion_walk": DatasetDefinition(
        "motion_walk", "synthetic/motion", "mocap35", False, 50, 49, 16, 7, 29, 6, 56
    ),
    "pems08": DatasetDefinition(
        "pems08", "real/pems08/pems08", "pems08", False, 60, 60, 199, 49, 170, 3, 548
    ),
}


class DatasetMismatchError(RuntimeError):
    """Raised when local processed arrays do not match the requested paper dataset."""


class IrregularArrayStorage:
    """Load one source split once so train and validation views share memory."""

    def __init__(self, data_root: str | Path, definition: DatasetDefinition, split: str) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.definition = definition
        self.split = split
        directory = self.data_root / definition.relative_directory
        suffix = definition.suffix
        self.loc = np.load(directory / f"loc_{split}_{suffix}.npy", allow_pickle=True)
        self.times = np.load(directory / f"times_{split}_{suffix}.npy", allow_pickle=True)
        self.edges = np.load(directory / f"edges_{split}_{suffix}.npy", allow_pickle=True)
        self.vel = (
            np.load(directory / f"vel_{split}_{suffix}.npy", allow_pickle=True)
            if definition.use_velocity
            else None
        )
        self._validate()

    def _validate(self) -> None:
        expected_count = (
            self.definition.expected_train
            if self.split == "train"
            else self.definition.expected_test
        )
        errors: list[str] = []
        if len(self.loc) != expected_count:
            errors.append(f"expected {expected_count} sequences, found {len(self.loc)}")
        if self.loc.ndim != 2 or self.loc.shape[1] != self.definition.expected_nodes:
            errors.append(
                f"expected {self.definition.expected_nodes} nodes, found shape {self.loc.shape}"
            )
        feature_dim = self.feature_array(0, 0).shape[-1]
        if feature_dim != self.definition.expected_features:
            errors.append(
                f"expected {self.definition.expected_features} features, found {feature_dim}"
            )
        if self.edges.shape != (
            expected_count,
            self.definition.expected_nodes,
            self.definition.expected_nodes,
        ):
            errors.append(f"unexpected adjacency shape {self.edges.shape}")
        if errors:
            raise DatasetMismatchError(
                f"{self.definition.name}/{self.split} does not match the paper data: "
                + "; ".join(errors)
            )

    def feature_array(self, sample_index: int, node_index: int) -> np.ndarray:
        location = np.asarray(self.loc[sample_index, node_index], dtype=np.float32)
        if self.vel is None:
            return location
        velocity = np.asarray(self.vel[sample_index, node_index], dtype=np.float32)
        return np.concatenate([location, velocity], axis=-1)

    def max_abs(self) -> float:
        value = 0.0
        for sample_index in range(len(self.loc)):
            for node_index in range(self.definition.expected_nodes):
                features = self.feature_array(sample_index, node_index)
                if features.size:
                    value = max(value, float(np.abs(features).max()))
        return value if value > 0.0 else 1.0


def _normalize_fixed_interval(
    raw_times: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    denominator = max(float(end - start), 1.0)
    return ((raw_times.astype(np.float32) - float(start)) / denominator).astype(np.float32)


def _physical_graph_and_types(
    raw_adjacency: np.ndarray,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_adjacency)
    if dataset == "springs":
        adjacency = (raw != 0).astype(np.float32)
        edge_type = np.where(raw != 0, 1, 0).astype(np.int64)
        # NRI treats no-spring as a relation on candidate off-diagonal pairs.
        np.fill_diagonal(edge_type, -1)
    elif dataset == "charged":
        # Both -1 and +1 are active interactions; source diagonal entries are retained.
        adjacency = (np.abs(raw) > 0).astype(np.float32)
        edge_type = np.where(raw < 0, 0, 1).astype(np.int64)
    else:
        adjacency = (raw != 0).astype(np.float32)
        edge_type = np.where(adjacency > 0, 0, -1).astype(np.int64)
    return adjacency, edge_type


def _place_on_union(
    node_times: list[np.ndarray],
    node_values: list[np.ndarray],
    num_nodes: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonempty = [times for times in node_times if times.size]
    if not nonempty:
        raise DatasetMismatchError("A sample has no timestamps in the requested task interval")
    union = np.unique(np.concatenate(nonempty)).astype(np.float32)
    values = np.zeros((len(union), num_nodes, feature_dim), dtype=np.float32)
    mask = np.zeros((len(union), num_nodes), dtype=np.bool_)
    index_by_time = {float(value): index for index, value in enumerate(union)}
    for node_index, (times, features) in enumerate(zip(node_times, node_values)):
        for local_index, raw_time in enumerate(times):
            union_index = index_by_time[float(raw_time)]
            values[union_index, node_index] = features[local_index]
            mask[union_index, node_index] = True
    return union, values, mask


class CSGODEDataset(Dataset[dict[str, Any]]):
    """A deterministic view over one source split using the standardized CSG batch."""

    def __init__(
        self,
        storage: IrregularArrayStorage,
        indices: Iterable[int],
        config: CSGODEConfig,
        *,
        logical_split: str,
        normalization_scale: float,
        communicability_cache: CommunicabilityCache | None = None,
    ) -> None:
        self.storage = storage
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.config = config
        self.logical_split = logical_split
        self.normalization_scale = float(normalization_scale)
        self.communicability_cache = communicability_cache or DEFAULT_COMMUNICABILITY_CACHE

    def __len__(self) -> int:
        return len(self.indices)

    def _periods(self) -> tuple[tuple[float, float], tuple[float, float]]:
        definition = self.storage.definition
        if self.config.task == "interp":
            period = (0.0, float(definition.context_steps - 1))
            return period, period
        if self.storage.split == "train":
            midpoint = definition.context_steps // 2
            return (
                (0.0, float(midpoint - 1)),
                (float(midpoint), float(definition.context_steps - 1)),
            )
        return (
            (0.0, float(definition.context_steps - 1)),
            (
                float(definition.context_steps),
                float(definition.context_steps + definition.future_steps - 1),
            ),
        )

    def _node_segments(
        self,
        source_index: int,
        node_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_times = np.asarray(self.storage.times[source_index, node_index], dtype=np.float32)
        raw_values = self.storage.feature_array(source_index, node_index)
        context_period, target_period = self._periods()
        context_selector = (raw_times >= context_period[0]) & (raw_times <= context_period[1])
        target_selector = (raw_times >= target_period[0]) & (raw_times <= target_period[1])
        context_times = raw_times[context_selector]
        context_values = raw_values[context_selector]
        target_times = raw_times[target_selector]
        target_values = raw_values[target_selector]

        if self.config.drop_initial_observation and context_times.size:
            context_times = context_times[1:]
            context_values = context_values[1:]
            if self.config.task == "interp":
                target_times = target_times[1:]
                target_values = target_values[1:]
        return context_times, context_values, target_times, target_values

    def __getitem__(self, item: int) -> dict[str, Any]:
        source_index = int(self.indices[item])
        definition = self.storage.definition
        num_nodes = definition.expected_nodes
        feature_dim = definition.expected_features
        context_period, target_period = self._periods()

        observed_times_by_node: list[np.ndarray] = []
        observed_values_by_node: list[np.ndarray] = []
        target_times_by_node: list[np.ndarray] = []
        target_values_by_node: list[np.ndarray] = []
        observed_raw_sets: list[set[float]] = []
        observation_counts = np.zeros(num_nodes, dtype=np.int64)

        for node_index in range(num_nodes):
            context_times, context_values, target_times, target_values = self._node_segments(
                source_index, node_index
            )
            if len(context_times) == 0:
                raise DatasetMismatchError(
                    f"sample {source_index}, node {node_index} has no context observations"
                )
            selected = deterministic_observation_indices(
                len(context_times),
                self.config.obs_ratio,
                global_seed=self.config.seed,
                split=self.logical_split,
                sample_index=source_index,
                node_index=node_index,
            )
            keep_count = len(selected)
            selected_times = context_times[selected]
            selected_values = context_values[selected] / self.normalization_scale
            observed_times_by_node.append(selected_times)
            observed_values_by_node.append(selected_values.astype(np.float32))
            target_times_by_node.append(target_times)
            target_values_by_node.append(
                (target_values / self.normalization_scale).astype(np.float32)
            )
            observed_raw_sets.append({float(value) for value in selected_times})
            observation_counts[node_index] = keep_count

        observed_raw_times, observed_values, observed_node_mask = _place_on_union(
            observed_times_by_node,
            observed_values_by_node,
            num_nodes,
            feature_dim,
        )
        target_raw_times, target_values, target_node_mask = _place_on_union(
            target_times_by_node,
            target_values_by_node,
            num_nodes,
            feature_dim,
        )
        encoder_times = _normalize_fixed_interval(
            observed_raw_times, context_period[0], context_period[1]
        )
        target_times = _normalize_fixed_interval(
            target_raw_times, target_period[0], target_period[1]
        )

        feature_mask = np.repeat(observed_node_mask[..., None], feature_dim, axis=-1)
        target_mask = np.repeat(target_node_mask[..., None], feature_dim, axis=-1)
        target_observed_node = np.zeros_like(target_node_mask)
        if self.config.task == "interp":
            for time_index, raw_time in enumerate(target_raw_times):
                for node_index in range(num_nodes):
                    target_observed_node[time_index, node_index] = (
                        float(raw_time) in observed_raw_sets[node_index]
                    )
        target_observed = np.repeat(target_observed_node[..., None], feature_dim, axis=-1)
        target_unobserved = target_mask & ~target_observed

        adjacency, edge_type = _physical_graph_and_types(
            self.storage.edges[source_index], definition.name
        )
        importance = self.communicability_cache.get(adjacency)
        return {
            "full_values": torch.from_numpy(target_values.copy()),
            "full_mask": torch.from_numpy(target_mask.copy()),
            "observed_values": torch.from_numpy(observed_values),
            "observed_mask": torch.from_numpy(observed_node_mask),
            "feature_mask": torch.from_numpy(feature_mask),
            "encoder_times": torch.from_numpy(encoder_times),
            "encoder_times_raw": torch.from_numpy(observed_raw_times),
            "target_times": torch.from_numpy(target_times),
            "target_times_raw": torch.from_numpy(target_raw_times),
            "target_values": torch.from_numpy(target_values),
            "target_mask": torch.from_numpy(target_mask),
            "target_observed_mask": torch.from_numpy(target_observed),
            "target_unobserved_mask": torch.from_numpy(target_unobserved),
            "original_adjacency": torch.from_numpy(adjacency),
            "edge_type": torch.from_numpy(edge_type),
            "information_importance": importance,
            "observation_counts": torch.from_numpy(observation_counts),
            "sample_index": torch.tensor(source_index, dtype=torch.long),
            "normalization_scale": torch.tensor(self.normalization_scale, dtype=torch.float32),
            "dataset": definition.name,
            "split": self.logical_split,
        }

    def observation_count_summary(self) -> dict[str, float]:
        counts: list[int] = []
        for source_index in self.indices:
            for node_index in range(self.storage.definition.expected_nodes):
                context_times, _, _, _ = self._node_segments(int(source_index), node_index)
                counts.append(max(1, int(len(context_times) * self.config.obs_ratio)))
        array = np.asarray(counts)
        return {
            "minimum": int(array.min()),
            "maximum": int(array.max()),
            "mean": float(array.mean()),
            "std": float(array.std()),
        }


def collate_csg_batches(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(samples)
    max_observed = max(sample["encoder_times"].shape[0] for sample in samples)
    max_target = max(sample["target_times"].shape[0] for sample in samples)
    num_nodes = samples[0]["observed_values"].shape[1]
    feature_dim = samples[0]["observed_values"].shape[2]

    batch: dict[str, Any] = {
        "full_values": torch.zeros(batch_size, max_target, num_nodes, feature_dim),
        "full_mask": torch.zeros(batch_size, max_target, num_nodes, feature_dim, dtype=torch.bool),
        "observed_values": torch.zeros(batch_size, max_observed, num_nodes, feature_dim),
        "observed_mask": torch.zeros(batch_size, max_observed, num_nodes, dtype=torch.bool),
        "feature_mask": torch.zeros(
            batch_size, max_observed, num_nodes, feature_dim, dtype=torch.bool
        ),
        "encoder_times": torch.zeros(batch_size, max_observed),
        "encoder_times_raw": torch.zeros(batch_size, max_observed),
        "time_valid_mask": torch.zeros(batch_size, max_observed, dtype=torch.bool),
        "target_times": torch.zeros(batch_size, max_target),
        "target_times_raw": torch.zeros(batch_size, max_target),
        "target_time_valid_mask": torch.zeros(batch_size, max_target, dtype=torch.bool),
        "target_values": torch.zeros(batch_size, max_target, num_nodes, feature_dim),
        "target_mask": torch.zeros(
            batch_size, max_target, num_nodes, feature_dim, dtype=torch.bool
        ),
        "target_observed_mask": torch.zeros(
            batch_size, max_target, num_nodes, feature_dim, dtype=torch.bool
        ),
        "target_unobserved_mask": torch.zeros(
            batch_size, max_target, num_nodes, feature_dim, dtype=torch.bool
        ),
        "original_adjacency": torch.stack([sample["original_adjacency"] for sample in samples]),
        "edge_type": torch.stack([sample["edge_type"] for sample in samples]),
        "information_importance": torch.stack(
            [sample["information_importance"] for sample in samples]
        ),
        "observation_counts": torch.stack([sample["observation_counts"] for sample in samples]),
        "sample_index": torch.stack([sample["sample_index"] for sample in samples]),
        "normalization_scale": torch.stack([sample["normalization_scale"] for sample in samples]),
        "dataset": [sample["dataset"] for sample in samples],
        "split": [sample["split"] for sample in samples],
    }
    for batch_index, sample in enumerate(samples):
        observed_length = sample["encoder_times"].shape[0]
        target_length = sample["target_times"].shape[0]
        for key in (
            "observed_values",
            "observed_mask",
            "feature_mask",
            "encoder_times",
            "encoder_times_raw",
        ):
            batch[key][batch_index, :observed_length] = sample[key]
        batch["time_valid_mask"][batch_index, :observed_length] = True
        for key in (
            "full_values",
            "full_mask",
            "target_times",
            "target_times_raw",
            "target_values",
            "target_mask",
            "target_observed_mask",
            "target_unobserved_mask",
        ):
            batch[key][batch_index, :target_length] = sample[key]
        batch["target_time_valid_mask"][batch_index, :target_length] = True
    return batch


@dataclass
class DatasetBundle:
    train: CSGODEDataset
    validation: CSGODEDataset
    test: CSGODEDataset
    audit: dict[str, Any]


def _storage_audit(storage: IrregularArrayStorage) -> dict[str, Any]:
    definition = storage.definition
    first_adjacency = np.asarray(storage.edges[0])
    edge_counts = np.count_nonzero(storage.edges, axis=(1, 2))
    diagonals = np.diagonal(storage.edges, axis1=1, axis2=2)
    diagonal_counts = np.count_nonzero(diagonals, axis=1)
    lengths = np.asarray(
        [
            len(np.asarray(storage.times[sample_index, node_index]))
            for sample_index in range(len(storage.times))
            for node_index in range(definition.expected_nodes)
        ]
    )
    return {
        "sequences": int(len(storage.loc)),
        "nodes": int(storage.loc.shape[1]),
        "features": int(storage.feature_array(0, 0).shape[-1]),
        "adjacency_shape": list(first_adjacency.shape),
        "edge_values": np.unique(storage.edges).tolist(),
        "first_graph_nonzero_entries": int(np.count_nonzero(first_adjacency)),
        "edge_entries_min": int(edge_counts.min()),
        "edge_entries_max": int(edge_counts.max()),
        "edge_entries_mean": float(edge_counts.mean()),
        "diagonal_values": np.unique(diagonals).tolist(),
        "diagonal_nonzero_min": int(diagonal_counts.min()),
        "diagonal_nonzero_max": int(diagonal_counts.max()),
        "all_graphs_symmetric": bool(np.all(storage.edges == storage.edges.transpose(0, 2, 1))),
        "fixed_topology": bool(np.all(storage.edges == storage.edges[0])),
        "timestamp_count_min": int(lengths.min()),
        "timestamp_count_max": int(lengths.max()),
        "timestamp_count_mean": float(lengths.mean()),
        "timestamp_min": float(min(np.min(np.asarray(value)) for value in storage.times.flat)),
        "timestamp_max": float(max(np.max(np.asarray(value)) for value in storage.times.flat)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pems_feature_semantics(data_root: str | Path) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    raw_path = data_root / "real/pems08/raw/data.npz"
    sidecar_path = data_root / "real/pems08/raw/feature_semantics.json"
    if not raw_path.exists():
        return {
            "paper_compatible": False,
            "status": "unverified",
            "reason": f"Raw feature source is unavailable at {raw_path}",
        }

    raw_sha256 = _sha256_file(raw_path)
    with np.load(raw_path) as archive:
        raw = np.asarray(archive["data"])
    if raw.ndim != 3 or raw.shape[1:] != (170, 3):
        return {
            "paper_compatible": False,
            "status": "mismatch",
            "raw_sha256": raw_sha256,
            "reason": f"Expected raw [T,170,3], found {list(raw.shape)}",
        }
    time_index = np.arange(raw.shape[0])
    expected_time_of_day = (time_index % 288) / 288.0
    expected_day_of_week = (time_index // 288) % 7
    time_of_day_match = bool(
        np.array_equal(raw[:, 0, 1], expected_time_of_day) and np.all(raw[:, :, 1] == raw[:, :1, 1])
    )
    day_of_week_match = bool(
        np.array_equal(raw[:, 0, 2], expected_day_of_week) and np.all(raw[:, :, 2] == raw[:, :1, 2])
    )
    statistics = [
        {
            "minimum": float(raw[..., index].min()),
            "maximum": float(raw[..., index].max()),
            "mean": float(raw[..., index].mean()),
            "std": float(raw[..., index].std()),
        }
        for index in range(3)
    ]
    if time_of_day_match and day_of_week_match:
        return {
            "paper_compatible": False,
            "status": "mismatch",
            "detected_features": ["traffic_flow", "time_of_day", "day_of_week"],
            "required_features": ["traffic_flow", "speed", "occupancy"],
            "channel_statistics": statistics,
            "raw_sha256": raw_sha256,
            "reason": (
                "Channels 1 and 2 exactly encode time-of-day and day-of-week, not the "
                "paper's speed and occupancy features."
            ),
        }

    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {
                "paper_compatible": False,
                "status": "invalid_sidecar",
                "required_features": ["traffic_flow", "speed", "occupancy"],
                "channel_statistics": statistics,
                "raw_sha256": raw_sha256,
                "sidecar": str(sidecar_path),
                "reason": f"Could not parse feature-semantics sidecar: {error}",
            }
        declared = sidecar.get("features")
        declared_sha256 = sidecar.get("raw_sha256")
        declared_source = sidecar.get("source")
        labels_match = declared == ["traffic_flow", "speed", "occupancy"]
        checksum_matches = declared_sha256 == raw_sha256
        source_present = isinstance(declared_source, str) and bool(declared_source.strip())
        compatible = labels_match and checksum_matches and source_present
        if not labels_match:
            reason = "The sidecar does not declare traffic_flow, speed, and occupancy in order."
        elif not checksum_matches:
            reason = "The sidecar checksum does not match the current raw data.npz file."
        elif not source_present:
            reason = "The sidecar must record a non-empty source identifier."
        else:
            reason = "Feature labels and raw-file checksum are verified by the sidecar."
        return {
            "paper_compatible": compatible,
            "status": "verified_by_sidecar" if compatible else "mismatch",
            "declared_features": declared,
            "required_features": ["traffic_flow", "speed", "occupancy"],
            "channel_statistics": statistics,
            "raw_sha256": raw_sha256,
            "declared_raw_sha256": declared_sha256,
            "declared_source": declared_source,
            "sidecar": str(sidecar_path),
            "reason": reason,
        }
    return {
        "paper_compatible": False,
        "status": "unverified",
        "required_features": ["traffic_flow", "speed", "occupancy"],
        "channel_statistics": statistics,
        "raw_sha256": raw_sha256,
        "reason": (
            "The temporal-covariate pattern was not detected, but no feature_semantics.json "
            "sidecar binds the three channel labels to this raw-file checksum."
        ),
    }


def build_datasets(
    config: CSGODEConfig,
    data_root: str | Path,
    *,
    communicability_cache: CommunicabilityCache | None = None,
) -> DatasetBundle:
    dataset = canonical_dataset_name(config.dataset)
    definition = DATASETS[dataset]
    if (
        config.num_nodes != definition.expected_nodes
        or config.input_dim != definition.expected_features
    ):
        raise DatasetMismatchError(
            f"Config requests N={config.num_nodes}, F={config.input_dim}, but {dataset} "
            f"requires N={definition.expected_nodes}, F={definition.expected_features}"
        )
    cache = communicability_cache or DEFAULT_COMMUNICABILITY_CACHE
    train_storage = IrregularArrayStorage(data_root, definition, "train")
    test_storage = IrregularArrayStorage(data_root, definition, "test")

    permutation = np.random.default_rng(config.seed).permutation(len(train_storage.loc))
    validation_count = max(1, int(round(len(permutation) * config.validation_fraction)))
    validation_indices = np.sort(permutation[:validation_count])
    train_indices = np.sort(permutation[validation_count:])

    train_scale = train_storage.max_abs()
    test_scale = (
        test_storage.max_abs()
        if config.normalization_mode == "paper_splitwise_maxabs"
        else train_scale
    )
    train = CSGODEDataset(
        train_storage,
        train_indices,
        config,
        logical_split="train",
        normalization_scale=train_scale,
        communicability_cache=cache,
    )
    validation = CSGODEDataset(
        train_storage,
        validation_indices,
        config,
        logical_split="validation",
        normalization_scale=train_scale,
        communicability_cache=cache,
    )
    test = CSGODEDataset(
        test_storage,
        np.arange(len(test_storage.loc)),
        config,
        logical_split="test",
        normalization_scale=test_scale,
        communicability_cache=cache,
    )
    audit = {
        "dataset": dataset,
        "definition": definition.__dict__,
        "train_source": _storage_audit(train_storage),
        "test_source": _storage_audit(test_storage),
        "training_examples_after_validation_split": len(train),
        "validation_examples": len(validation),
        "train_maxabs": train_scale,
        "test_applied_maxabs": test_scale,
        "normalization_mode": config.normalization_mode,
        "mask_algorithm": MASK_ALGORITHM_VERSION,
        "motion_feature_convention": (
            "six pose channels from loc; derivative vel file intentionally not concatenated"
            if dataset == "motion_walk"
            else None
        ),
        "pems_global_split_verification": (
            "Processed windows contain local timestamps; global non-overlap is established "
            "by the preprocessing script, not recoverable from these arrays alone."
            if dataset == "pems08"
            else None
        ),
    }
    if dataset == "pems08":
        audit["feature_semantics"] = audit_pems_feature_semantics(data_root)
        audit["paper_compatible"] = audit["feature_semantics"]["paper_compatible"]
    else:
        audit["paper_compatible"] = True
    return DatasetBundle(train, validation, test, audit)


def make_loaders(
    bundle: DatasetBundle,
    config: CSGODEConfig,
    *,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_csg_batches,
        "pin_memory": torch.cuda.is_available(),
    }
    train = DataLoader(bundle.train, shuffle=True, generator=generator, **common)
    validation = DataLoader(bundle.validation, shuffle=False, **common)
    test = DataLoader(bundle.test, shuffle=False, **common)
    return train, validation, test


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }

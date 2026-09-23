"""Configuration and paper hyperparameters for CSG-ODE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


DATASET_ALIASES = {
    "spring": "springs",
    "springs5": "springs",
    "charged5": "charged",
    "motion": "motion_walk",
    "motion-walk": "motion_walk",
    "mocap35": "motion_walk",
    "pems": "pems08",
}


PAPER_DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "springs": {
        "num_nodes": 5,
        "input_dim": 4,
        "batch_size": 256,
        "observation_embedding_dim": 64,
        "node_parameter_dim": 32,
        "augment_dim": 64,
        "subnetwork_width": 128,
        "num_relations": 2,
    },
    "charged": {
        "num_nodes": 5,
        "input_dim": 4,
        "batch_size": 256,
        "observation_embedding_dim": 64,
        "node_parameter_dim": 32,
        "augment_dim": 64,
        "subnetwork_width": 128,
        "num_relations": 2,
    },
    "motion_walk": {
        "num_nodes": 29,
        "input_dim": 6,
        "batch_size": 8,
        "observation_embedding_dim": 64,
        "node_parameter_dim": 32,
        "augment_dim": 64,
        "subnetwork_width": 128,
        "num_relations": 1,
    },
    "pems08": {
        "num_nodes": 170,
        "input_dim": 3,
        "batch_size": 4,
        "observation_embedding_dim": 16,
        "node_parameter_dim": 8,
        "augment_dim": 0,
        "subnetwork_width": 64,
        "num_relations": 1,
    },
}


@dataclass(frozen=True)
class CSGODEConfig:
    """Resolved CSG-ODE settings saved verbatim in every checkpoint."""

    model: str = "csg_ode"
    dataset: str = "springs"
    task: str = "interp"
    obs_ratio: float = 0.6
    seed: int = 1991

    num_nodes: int = 5
    input_dim: int = 4
    observation_embedding_dim: int = 64
    node_parameter_dim: int = 32
    hidden_dim: int = 16
    latent_dim: int = 16
    augment_dim: int = 64
    adaptive_adj_dim: int | None = None

    alpha: float = 0.5
    density_activation: str = "sigmoid"
    clamp_density_factor: bool = False
    matrix_init: str = "zero"
    embedding_activation: str = "gelu"
    carry_unobserved_hidden: bool = True

    num_subnetworks: int = 2
    subnetwork_width: int = 128
    subnetwork_depth: int = 1
    dynamics_activation: str = "tanh"
    control_init: str = "zero"
    num_relations: int = 1

    solver: str = "euler"
    rtol: float = 1e-3
    atol: float = 1e-4
    obsrv_std: float = 0.01
    n_traj_samples: int = 3

    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    dropout: float = 0.2
    grad_clip: float = 10.0
    validation_fraction: float = 0.1
    normalization_mode: str = "paper_splitwise_maxabs"
    observation_protocol: str = "lgode_second_stage"
    drop_initial_observation: bool = True
    kl_schedule: str = "lgode_per_epoch"
    kl_wait_batches: int = 10
    kl_decay: float = 0.99

    ablation_aq: bool = False
    ablation_no_ei: bool = False
    ablation_no_g: bool = False
    ablation_no_ni: bool = False

    def __post_init__(self) -> None:
        dataset = DATASET_ALIASES.get(self.dataset, self.dataset)
        object.__setattr__(self, "dataset", dataset)
        if self.model != "csg_ode":
            raise ValueError("model must be 'csg_ode'")
        if dataset not in PAPER_DATASET_DEFAULTS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        if self.task not in {"interp", "extrap"}:
            raise ValueError("task must be 'interp' or 'extrap'")
        if not 0.0 < self.obs_ratio <= 1.0:
            raise ValueError("obs_ratio must be in (0, 1]")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        positive_dimensions = {
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
            "observation_embedding_dim": self.observation_embedding_dim,
            "node_parameter_dim": self.node_parameter_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_subnetworks": self.num_subnetworks,
            "subnetwork_width": self.subnetwork_width,
            "num_relations": self.num_relations,
            "n_traj_samples": self.n_traj_samples,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }
        invalid_dimensions = [name for name, value in positive_dimensions.items() if value <= 0]
        if invalid_dimensions:
            raise ValueError(f"These settings must be positive: {', '.join(invalid_dimensions)}")
        if self.augment_dim < 0 or self.subnetwork_depth < 0:
            raise ValueError("augment_dim and subnetwork_depth must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.obsrv_std <= 0 or self.grad_clip <= 0:
            raise ValueError("obsrv_std and grad_clip must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("rtol and atol must be positive")
        if self.kl_wait_batches < 0 or not 0.0 < self.kl_decay <= 1.0:
            raise ValueError("Invalid KL schedule parameters")
        if self.resolved_adaptive_adj_dim <= 0:
            raise ValueError("adaptive_adj_dim must be positive")
        if self.density_activation not in {"sigmoid", "tanh", "identity"}:
            raise ValueError("Unsupported density activation")
        if self.matrix_init not in {"zero", "xavier"}:
            raise ValueError("matrix_init must be zero or xavier")
        if self.embedding_activation not in {"gelu", "relu"}:
            raise ValueError("embedding_activation must be gelu or relu")
        if self.dynamics_activation not in {"tanh", "relu"}:
            raise ValueError("dynamics_activation must be tanh or relu")
        if self.control_init not in {"zero", "gnn"}:
            raise ValueError("control_init must be zero or gnn")
        if self.solver not in {"euler", "dopri5"}:
            raise ValueError("solver must be euler or dopri5")
        if self.normalization_mode not in {
            "paper_splitwise_maxabs",
            "train_maxabs",
        }:
            raise ValueError("Unsupported normalization mode")
        if self.observation_protocol != "lgode_second_stage":
            raise ValueError("Only the verified LG-ODE observation protocol is supported")
        if self.kl_schedule not in {"lgode_per_epoch", "global", "constant"}:
            raise ValueError("Unsupported KL schedule")

    @property
    def resolved_adaptive_adj_dim(self) -> int:
        return self.input_dim if self.adaptive_adj_dim is None else self.adaptive_adj_dim

    @property
    def dynamics_dim(self) -> int:
        return self.latent_dim + self.augment_dim

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["adaptive_adj_dim"] = self.resolved_adaptive_adj_dim
        result["dynamics_dim"] = self.dynamics_dim
        return result

    def with_overrides(self, **overrides: Any) -> "CSGODEConfig":
        valid = {field.name for field in fields(self)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise KeyError(f"Unknown CSG-ODE settings: {', '.join(unknown)}")
        return replace(self, **overrides)


def canonical_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def paper_config(dataset: str, **overrides: Any) -> CSGODEConfig:
    """Create a config with the ICML paper's dataset-specific values."""

    dataset = canonical_dataset_name(dataset)
    if dataset not in PAPER_DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    values = {"dataset": dataset, **PAPER_DATASET_DEFAULTS[dataset], **overrides}
    return CSGODEConfig(**values)


def load_config(path: str | Path, **overrides: Any) -> CSGODEConfig:
    """Load a flat YAML config and apply explicit command-line overrides."""

    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise TypeError("CSG-ODE config must be a YAML mapping")
    dataset = canonical_dataset_name(
        str(overrides.get("dataset", values.get("dataset", "springs")))
    )
    base = paper_config(dataset)
    merged = {**values, **{key: value for key, value in overrides.items() if value is not None}}
    return base.with_overrides(**merged)

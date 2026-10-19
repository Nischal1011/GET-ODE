from __future__ import annotations

import torch

from lib.baselines.csg_ode.config import paper_config
from lib.baselines.csg_ode.encoder import CSGEncoder
from lib.baselines.csg_ode.model import CSGODE


def encoder_config(**overrides):
    values = {
        "num_nodes": 3,
        "input_dim": 2,
        "observation_embedding_dim": 5,
        "node_parameter_dim": 4,
        "hidden_dim": 6,
        "latent_dim": 3,
        "augment_dim": 0,
        "subnetwork_width": 8,
        "batch_size": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return paper_config("springs", **values)


def synthetic_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    values = torch.randn(2, 4, 3, 2)
    observed = torch.tensor(
        [
            [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]],
        ],
        dtype=torch.bool,
    )
    adjacency = torch.ones(2, 3, 3) - torch.eye(3).unsqueeze(0)
    return {
        "observed_values": values * observed.unsqueeze(-1),
        "observed_mask": observed,
        "feature_mask": observed.unsqueeze(-1).expand(-1, -1, -1, 2),
        "encoder_times": torch.tensor([[0.0, 0.3, 0.7, 1.0]]).expand(2, -1),
        "time_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "original_adjacency": adjacency,
        "information_importance": torch.zeros_like(adjacency),
    }


def test_mean_embedding_uses_observed_values_only() -> None:
    encoder = CSGEncoder(encoder_config()).eval()
    batch = synthetic_batch()
    first = encoder(batch)
    modified = {key: value.clone() for key, value in batch.items()}
    missing = ~modified["observed_mask"]
    modified["observed_values"][missing] = 10000.0
    second = encoder(modified)
    torch.testing.assert_close(first["mean_embedding"], second["mean_embedding"])
    torch.testing.assert_close(first["node_representation"], second["node_representation"])


def test_encoder_shapes_positive_std_and_reparameterization() -> None:
    config = encoder_config()
    batch = synthetic_batch()
    encoder = CSGEncoder(config)
    result = encoder(batch)
    assert result["mean"].shape == (2, 3, 3)
    assert result["std"].shape == (2, 3, 3)
    assert result["kl_per_node"].shape == (2, 3)
    assert result["std"].gt(0).all()

    batch.update(
        {
            "target_times": torch.tensor([[0.0, 0.5, 1.0]]).expand(2, -1),
            "target_time_valid_mask": torch.ones(2, 3, dtype=torch.bool),
            "edge_type": torch.where(batch["original_adjacency"] > 0, 0, -1).long(),
        }
    )
    model = CSGODE(config).eval()
    output = model(batch, n_traj_samples=2)
    assert output["predictions"].shape == (2, 2, 3, 3, 2)


def test_unobserved_current_node_carries_hidden_state() -> None:
    encoder = CSGEncoder(encoder_config())
    cell = encoder.recurrent_cell
    values = torch.randn(1, 3, 5)
    hidden = torch.randn(1, 3, 6)
    graph = torch.softmax(torch.randn(1, 3, 3), dim=-1)
    representation = torch.randn(1, 3, 4)
    observed = torch.tensor([[True, False, True]])
    next_hidden = cell(
        values,
        hidden,
        graph,
        representation,
        observed,
        carry_unobserved=True,
    )
    torch.testing.assert_close(next_hidden[:, 1], hidden[:, 1])


def test_no_ei_ablation_reduces_graph_mix_to_adaptive_graph() -> None:
    config = encoder_config(ablation_no_ei=True, matrix_init="xavier")
    encoder = CSGEncoder(config)
    result = encoder(synthetic_batch())
    expected = result["adaptive_graph"].unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(result["graph_mix"], expected)


def test_aq_ablation_uses_free_sample_independent_node_parameters() -> None:
    encoder = CSGEncoder(encoder_config(ablation_aq=True)).eval()
    batch = synthetic_batch()
    result = encoder(batch)
    expected = encoder.free_node_representation.unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(result["node_representation"], expected)
    torch.testing.assert_close(result["node_representation"][0], result["node_representation"][1])


def test_gradients_reach_adaptive_and_node_parameter_generators() -> None:
    encoder = CSGEncoder(encoder_config(matrix_init="xavier"))
    with torch.no_grad():
        encoder.adaptive_graph.source_embedding.fill_(0.5)
        encoder.adaptive_graph.target_embedding.copy_(
            torch.arange(6, dtype=torch.float32).reshape(3, 2) / 10
        )
    result = encoder(synthetic_batch())
    loss = result["mean"].square().sum() + result["std"].sum()
    loss.backward()
    assert encoder.adaptive_graph.source_embedding.grad is not None
    assert encoder.adaptive_graph.target_embedding.grad is not None
    assert encoder.node_representation[0].weight.grad is not None
    assert encoder.recurrent_cell.gates.weight_pool.grad is not None
    assert encoder.recurrent_cell.candidate.weight_pool.grad is not None

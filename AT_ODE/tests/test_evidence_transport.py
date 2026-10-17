from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch_geometric.data import Batch, Data


AT_ODE_ROOT = Path(__file__).resolve().parents[1]
if str(AT_ODE_ROOT) not in sys.path:
    sys.path.insert(0, str(AT_ODE_ROOT))

from lib.at_transport_posterior import ATTransportPosterior
from lib.create_latent_ode_model import create_LatentODE_model
from lib.evidence_transport_posterior import (
    EvidenceTransportPosterior,
    RelationalEvidenceHead,
    StaticRelationPosterior,
)
from lib.gnn_models import NRIConv
from lib.new_dataLoader import ParseData
from lib import utils


def relation_matrices(n_atoms, order=None, device="cpu"):
    receiver, sender = np.where(np.ones((n_atoms, n_atoms)) - np.eye(n_atoms))
    receiver = torch.tensor(receiver, dtype=torch.long, device=device)
    sender = torch.tensor(sender, dtype=torch.long, device=device)
    if order is not None:
        order = torch.tensor(order, dtype=torch.long, device=device)
        receiver = receiver[order]
        sender = sender[order]
    return (
        F.one_hot(sender, n_atoms).float(),
        F.one_hot(receiver, n_atoms).float(),
    )


class FixedEvidenceHead(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, h_src, h_dst, delta_t, mask_src, mask_dst):
        return self.values[: h_src.size(0)].to(h_src.device)


def event_batch(graph_specs, event_dim=4):
    data_list = []
    for spec in graph_specs:
        object_id = torch.tensor(spec["object_id"], dtype=torch.long)
        pos = torch.tensor(spec["pos"], dtype=torch.float32)
        edge_index = torch.tensor(spec["edge_index"], dtype=torch.long)
        src, dst = edge_index
        data_list.append(
            Data(
                x=torch.zeros(object_id.numel(), 2),
                edge_index=edge_index,
                edge_attr=pos[src] - pos[dst],
                edge_same=(object_id[src] == object_id[dst]).float(),
                pos=pos,
                object_id=object_id,
                y=torch.bincount(object_id, minlength=2),
            )
        )
    batch = Batch.from_data_list(data_list)
    batch.event_embeddings = torch.randn(
        batch.num_nodes, event_dim, requires_grad=True
    )
    return batch


def make_posterior(
    head,
    *,
    ablation="full",
    prior_mode="uniform",
    decay_init=1e-6,
    velocity=1.0,
    K_lag=3,
    lag_max=1.0,
    order=None,
    device="cpu",
    tau=1.0,
    hard=False,
):
    rel_send, rel_rec = relation_matrices(2, order=order, device=device)
    return EvidenceTransportPosterior(
        n_atoms=2,
        edge_types=2,
        rel_send=rel_send,
        rel_rec=rel_rec,
        evidence_head=head,
        K_lag=K_lag,
        lag_max=lag_max,
        transport_velocity=velocity,
        decay_init=decay_init,
        age_kernel="uniform",
        prior_mode=prior_mode,
        edge_prior_p=0.9,
        prior_strength=1.0,
        tau_gumbel=tau,
        hard_gumbel=hard,
        ablation=ablation,
    ).to(device)


def one_cross_edge_batch():
    batch = event_batch(
        [{"object_id": [0, 1], "pos": [0.0, 0.0], "edge_index": [[0], [1]]}]
    )
    batch.physical_edge_labels = torch.tensor([[0, 1]])
    return batch


def test_loader_adds_local_object_ids_without_reordering_nodes():
    args = SimpleNamespace(
        random_seed=1,
        total_ode_step=60,
        cutting_edge=True,
        extrap_num=40,
    )
    loader = ParseData("unused", args, suffix="_springs5", mode="interp")
    loader.num_atoms = 2
    loader.has_vel = False
    loader.sample_percent = 0.6
    loc = [np.zeros((2, 2)), np.ones((3, 2))]
    times = [np.array([0.0, 0.1]), np.array([0.0, 0.2, 0.3])]
    graph, _, _ = loader.transfer_one_graph(
        loc,
        None,
        np.array([[0, 1], [1, 0]]),
        times,
    )
    assert graph.object_id.tolist() == [0, 0, 1, 1, 1]
    assert torch.allclose(graph.pos, torch.tensor([0.0, 0.1, 0.0, 0.2, 0.3]))


def test_evidence_head_is_nonnegative_and_can_be_near_zero():
    head = RelationalEvidenceHead(4, 2, hidden_dim=8, init_bias=-20.0)
    nn.init.zeros_(head.hidden.weight)
    nn.init.zeros_(head.hidden.bias)
    nn.init.zeros_(head.output.weight)
    inputs = torch.zeros(5, 4)
    evidence = head(inputs, inputs, torch.zeros(5), torch.ones(5), torch.ones(5))
    assert torch.all(evidence >= 0)
    assert evidence.max() < 1e-7


def test_zero_evidence_recovers_graph_prior_and_normalizes_per_edge():
    batch = one_cross_edge_batch()
    posterior = make_posterior(
        FixedEvidenceHead([[0.0, 0.0]]), prior_mode="graph"
    )
    _, extras = posterior.build(batch, torch.tensor([0.0, 0.25]), sample=False)
    expected = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    assert torch.allclose(extras["q_probs_grid"][0], expected)
    assert torch.allclose(
        extras["q_probs_grid"].sum(-1),
        torch.ones_like(extras["q_probs_grid"][..., 0]),
    )


def test_evidence_changes_only_its_relation_and_physical_edge():
    batch = one_cross_edge_batch()
    posterior = make_posterior(FixedEvidenceHead([[0.0, 6.0]]))
    _, extras = posterior.build(batch, torch.tensor([0.0]), sample=False)
    q = extras["q_probs_grid"][0, 0]
    mapped_edge = extras["physical_edge_indices"].item()
    unrelated_edge = 1 - mapped_edge
    assert q[mapped_edge, 1] > 0.5
    assert torch.allclose(q[unrelated_edge], torch.tensor([0.5, 0.5]))


def test_event_mapping_uses_generator_order_not_row_major_assumption():
    batch = one_cross_edge_batch()
    posterior = make_posterior(
        FixedEvidenceHead([[1.0, 0.0]]), order=[1, 0]
    )
    _, extras = posterior.build(batch, torch.tensor([0.0]), sample=False)
    assert extras["physical_pair_indices"].tolist() == [[0, 0, 1]]
    assert extras["physical_edge_indices"].item() == 0


def test_evidence_source_time_is_receiver_event_time():
    batch = event_batch(
        [{"object_id": [0, 1], "pos": [0.1, 0.4], "edge_index": [[0], [1]]}]
    )
    batch.physical_edge_labels = torch.zeros(1, 2, dtype=torch.long)
    posterior = make_posterior(FixedEvidenceHead([[1.0, 0.0]]))
    _, extras = posterior.build(batch, torch.tensor([0.4]), sample=False)
    assert torch.allclose(extras["source_event_times"], torch.tensor([0.4]))


def test_batch_isolation():
    batch = event_batch(
        [
            {"object_id": [0, 1], "pos": [0.0, 0.0], "edge_index": [[0], [1]]},
            {"object_id": [0, 1], "pos": [0.0, 0.0], "edge_index": [[0], [1]]},
        ]
    )
    batch.physical_edge_labels = torch.zeros(2, 2, dtype=torch.long)
    posterior = make_posterior(FixedEvidenceHead([[0.0, 8.0], [8.0, 0.0]]))
    _, extras = posterior.build(batch, torch.tensor([0.0]), sample=False)
    edge_id = extras["physical_edge_indices"][0]
    q = extras["q_probs_grid"][0]
    assert q[0, edge_id, 1] > 0.5
    assert q[1, edge_id, 0] > 0.5
    assert torch.allclose(q[:, 1 - edge_id], torch.full((2, 2), 0.5))


def test_same_object_event_edges_are_filtered():
    batch = event_batch(
        [{"object_id": [0, 0, 1], "pos": [0.0, 0.1, 0.1], "edge_index": [[0, 0], [1, 2]]}]
    )
    batch.physical_edge_labels = torch.zeros(1, 2, dtype=torch.long)
    posterior = make_posterior(FixedEvidenceHead([[3.0, 0.0]]))
    _, extras = posterior.build(batch, torch.tensor([0.1]), sample=False)
    assert extras["event_evidence"].shape == (1, 2)
    assert extras["physical_pair_indices"].tolist() == [[0, 0, 1]]


def test_age_moves_toward_older_bins():
    posterior = make_posterior(FixedEvidenceHead([]), decay_init=1e-6)
    state = torch.zeros(1, 2, 2, 3)
    state[0, 0, 0, 0] = 1.0
    advanced = posterior._advance_state(state, 0.5)
    assert advanced[0, 0, 0, 1] > 0.999
    assert advanced[0, 0, 0, 0] < 1e-6


def test_relation_specific_decay_reduces_total_concentration():
    posterior = make_posterior(
        FixedEvidenceHead([]), decay_init=0.7, velocity=0.0
    )
    state = torch.ones(1, 2, 2, 3)
    advanced = posterior._advance_state(state, 0.2)
    expected = state.sum() * torch.exp(torch.tensor(-0.7 * 0.2))
    assert torch.allclose(advanced.sum(), expected, atol=1e-5)


def test_oldest_lag_has_open_outflow():
    posterior = make_posterior(FixedEvidenceHead([]), decay_init=1e-6)
    state = torch.zeros(1, 2, 2, 3)
    state[0, 0, 0, 0] = 1.0
    advanced = posterior._advance_state(state, 1.5)
    assert advanced.sum() < 1e-6


def test_sample_false_ignores_gumbel_configuration():
    batch = one_cross_edge_batch()
    head_a = FixedEvidenceHead([[1.0, 3.0]])
    head_b = FixedEvidenceHead([[1.0, 3.0]])
    posterior_a = make_posterior(head_a, tau=0.01, hard=True)
    posterior_b = make_posterior(head_b, tau=20.0, hard=False)
    provider_a, extras_a = posterior_a.build(batch, torch.tensor([0.0, 0.5]), sample=False)
    provider_b, extras_b = posterior_b.build(batch, torch.tensor([0.0, 0.5]), sample=False)
    assert torch.allclose(extras_a["q_probs_grid"], extras_b["q_probs_grid"])
    assert torch.allclose(provider_a(torch.tensor(0.25)), provider_b(torch.tensor(0.25)))


def configure_nri(inactive_relation_index):
    conv = NRIConv(2, 2, dropout=0.0)
    rel_send, rel_rec = relation_matrices(2)
    conv.rel_send = rel_send
    conv.rel_rec = rel_rec
    conv.inactive_relation_index = inactive_relation_index
    conv.rel_type = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    for layer in [conv.msg_fc1[0], conv.msg_fc2[0]]:
        nn.init.ones_(layer.weight)
        nn.init.ones_(layer.bias)
    return conv


def test_charged_relation_zero_produces_a_message():
    conv = configure_nri(inactive_relation_index=None)
    conv(torch.ones(1, 2, 2))
    assert conv.last_message_norm > 0


def test_springs_no_spring_relation_is_inactive():
    conv = configure_nri(inactive_relation_index=0)
    conv(torch.ones(1, 2, 2))
    assert conv.last_message_norm == 0


def test_provider_tiles_for_multiple_trajectory_samples():
    conv = configure_nri(inactive_relation_index=None)
    conv.rel_type_provider = lambda _: torch.tensor(
        [[[0.25, 0.75], [0.75, 0.25]]]
    )
    output = conv(torch.ones(3, 2, 2), t_local=torch.tensor(0.2))
    assert output.shape == (3, 2, 2)


def test_gradients_reach_evidence_head_and_event_embeddings():
    batch = one_cross_edge_batch()
    head = RelationalEvidenceHead(4, 2, hidden_dim=8, init_bias=-2.0)
    posterior = make_posterior(head)
    _, extras = posterior.build(batch, torch.tensor([0.0]), sample=False)
    loss = -extras["q_probs_grid"][0, 0, :, 1].sum()
    loss.backward()
    assert torch.isfinite(head.gradient_norm())
    assert head.gradient_norm() > 0
    assert batch.event_embeddings.grad is not None
    assert batch.event_embeddings.grad.abs().sum() > 0


def test_legacy_attention_mapping_is_backward_compatible():
    batch = event_batch(
        [{"object_id": [0, 0, 1, 1], "pos": [0.0, 0.1, 0.0, 0.1], "edge_index": [[0, 2], [3, 1]]}]
    )
    batch.edge_attn = torch.tensor([[0.3], [0.7]])
    legacy_batch = batch.clone()
    del legacy_batch.object_id
    posterior_explicit = ATTransportPosterior(n_atoms=2, K_lag=3)
    posterior_legacy = ATTransportPosterior(n_atoms=2, K_lag=3)
    posterior_legacy.load_state_dict(posterior_explicit.state_dict())
    _, explicit = posterior_explicit.build(batch, torch.tensor([0.0, 0.5]), sample=False)
    _, legacy = posterior_legacy.build(legacy_batch, torch.tensor([0.0, 0.5]), sample=False)
    assert torch.allclose(explicit["q_probs_grid"], legacy["q_probs_grid"])


def tiny_model_args():
    return SimpleNamespace(
        latents=4,
        rec_dims=8,
        ode_dims=8,
        n_heads=1,
        rec_layers=1,
        dropout=0.0,
        z0_encoder="GTrans",
        rec_attention="add",
        augment_dim=0,
        gen_layers=1,
        odenet="NRI",
        solver="rk4",
        n_balls=2,
        edge_types=2,
        data="spring",
        edge_posterior_source="evidence",
        evidence_hidden_dim=8,
        evidence_init_bias=-2.0,
        evidence_decay_init=0.2,
        edge_prior_strength=1.0,
        age_kernel="uniform",
        lag_max=1.0,
        evidence_l1_coef=0.0,
        evidence_ablation="full",
        K_lag=4,
        attn_v=1.0,
        gumbel_tau=1.0,
        gumbel_hard=False,
        edge_prior_mode="graph",
        edge_prior_p=0.9,
        kl_edge_coef=1.0,
    )


@pytest.mark.parametrize(
    ("source", "expected_type"),
    [
        ("attention", ATTransportPosterior),
        ("constant", ATTransportPosterior),
        ("topology", StaticRelationPosterior),
        ("none", type(None)),
    ],
)
def test_factory_preserves_all_posterior_source_modes(source, expected_type):
    args = tiny_model_args()
    args.edge_posterior_source = source
    model = create_LatentODE_model(
        args,
        4,
        Normal(torch.tensor([0.0]), torch.tensor([1.0])),
        torch.tensor([0.01]),
        torch.device("cpu"),
    )
    assert isinstance(model.edge_posterior, expected_type)
    if source == "constant":
        assert model.edge_posterior.source_mode == "constant"


def test_legacy_ddp_attention_checkpoint_loads_without_module_wrapper(tmp_path):
    args = tiny_model_args()
    args.edge_posterior_source = "attention"
    prior = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    source_model = create_LatentODE_model(
        args, 4, prior, torch.tensor([0.01]), torch.device("cpu")
    )
    checkpoint = {
        "epoch": 3,
        "best_val_loss": 0.25,
        "model": {
            f"module.{key}": value for key, value in source_model.state_dict().items()
        },
        "optimizer": None,
        "scheduler": None,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": None,
            "numpy": np.random.get_state(),
        },
    }
    checkpoint_path = tmp_path / "legacy_attention.ckpt"
    torch.save(checkpoint, checkpoint_path)
    target_model = create_LatentODE_model(
        args, 4, prior, torch.tensor([0.01]), torch.device("cpu")
    )
    start_epoch, best_loss = utils.resume_from_pf(
        checkpoint_path, target_model, device="cpu"
    )
    assert start_epoch == 4
    assert best_loss == 0.25
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, target_model.state_dict()[key])


def test_end_to_end_loss_is_finite_and_evidence_gradient_is_nonzero():
    torch.manual_seed(7)
    data = Data(
        x=torch.randn(4, 4),
        object_id=torch.tensor([0, 0, 1, 1]),
        pos=torch.tensor([0.0, 0.1, 0.0, 0.1]),
        edge_index=torch.tensor([[0, 2, 0, 2], [1, 3, 3, 1]]),
        edge_attr=torch.full((4,), -0.1),
        edge_same=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        y=torch.tensor([2, 2]),
    )
    encoder_batch = Batch.from_data_list([data])
    decoder_batch = {
        "raw": torch.randn(2, 2, 4),
        "time_steps": torch.tensor([0.0, 0.1]),
        "mask": torch.ones(2, 2, 4),
    }
    graph_batch = torch.tensor([[1, 1]])
    prior = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    args = tiny_model_args()
    args.kl_edge_coef = 0.0
    model = create_LatentODE_model(
        args, 4, prior, torch.tensor([0.01]), torch.device("cpu")
    )
    assert any(
        key.startswith("edge_posterior.evidence_head")
        for key in model.state_dict()
    )
    assert any(
        parameter is model.edge_posterior.evidence_head.output.weight
        for parameter in model.parameters()
    )
    result = model(
        encoder_batch,
        decoder_batch,
        graph_batch,
        n_traj_samples=1,
        kl_coef=0.0,
    )
    result["loss"].backward()
    grad_norm = model.edge_posterior.evidence_head.gradient_norm()
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(grad_norm)
    assert grad_norm > 0
    assert np.isfinite(result["kl_edge"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_posterior_has_no_nans():
    device = torch.device("cuda")
    batch = one_cross_edge_batch().to(device)
    batch.event_embeddings = batch.event_embeddings.detach().to(device).requires_grad_(True)
    head = RelationalEvidenceHead(4, 2, hidden_dim=8).to(device)
    posterior = make_posterior(head, device=device)
    with torch.amp.autocast("cuda"):
        _, extras = posterior.build(batch, torch.tensor([0.0, 0.25], device=device), sample=False)
        loss = extras["q_probs_grid"][..., 1].mean()
    loss.backward()
    assert all(
        torch.isfinite(extras[key]).all()
        for key in ["event_evidence", "posterior_concentration_grid", "q_probs_grid"]
    )

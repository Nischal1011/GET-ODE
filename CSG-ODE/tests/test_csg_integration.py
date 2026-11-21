from __future__ import annotations

import os

import pytest
import torch

from conftest import DATA_ROOT
from lib.baselines.csg_ode.checkpoint import load_checkpoint, save_checkpoint
from lib.baselines.csg_ode.config import paper_config
from lib.baselines.csg_ode.data_adapter import build_datasets, collate_csg_batches
from lib.baselines.csg_ode.model import CSGODE
from run_csg_ode import ensure_run_directory_is_safe


def reduced_config(dataset: str, **overrides):
    values = {
        "observation_embedding_dim": 8,
        "node_parameter_dim": 4,
        "hidden_dim": 4,
        "latent_dim": 4,
        "augment_dim": 0,
        "subnetwork_width": 8,
        "dropout": 0.0,
        "n_traj_samples": 1,
        "batch_size": 1,
    }
    values.update(overrides)
    return paper_config(dataset, **values)


@pytest.mark.parametrize("dataset", ["springs", "charged", "motion_walk", "pems08"])
@pytest.mark.parametrize("task", ["interp", "extrap"])
def test_real_dataset_forward_backward_on_cpu(dataset: str, task: str) -> None:
    config = reduced_config(dataset, task=task)
    bundle = build_datasets(config, DATA_ROOT)
    batch = collate_csg_batches([bundle.train[0]])
    model = CSGODE(config).cpu()
    results = model.compute_all_losses(batch)
    results["loss"].backward()
    assert torch.isfinite(results["loss"])
    assert results["predictions"].shape[3:] == (config.num_nodes, config.input_dim)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_checkpoint_round_trip_gives_identical_mean_predictions(tmp_path) -> None:
    config = reduced_config("springs")
    bundle = build_datasets(config, DATA_ROOT)
    batch = collate_csg_batches([bundle.train[0]])
    first = CSGODE(config).eval()
    optimizer = torch.optim.Adam(first.parameters(), lr=config.learning_rate)
    expected = first(batch, sample_posterior=False)["predictions"].detach()
    path = tmp_path / "model.ckpt"
    save_checkpoint(
        path,
        first,
        optimizer,
        epoch=2,
        best_validation_mse=0.1,
        config=config,
        dataset_audit={},
    )
    second = CSGODE(config).eval()
    load_checkpoint(path, second, restore_rng=False)
    actual = second(batch, sample_posterior=False)["predictions"].detach()
    torch.testing.assert_close(actual, expected)


def test_existing_run_state_requires_an_explicit_matching_resume(tmp_path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "history.json").write_text("[]", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to mix training histories"):
        ensure_run_directory_is_safe(run_directory, None)

    checkpoint = run_directory / "last.ckpt"
    checkpoint.touch()
    ensure_run_directory_is_safe(run_directory, checkpoint)

    external = tmp_path / "other" / "last.ckpt"
    external.parent.mkdir()
    external.touch()
    with pytest.raises(FileExistsError, match="already has state"):
        ensure_run_directory_is_safe(run_directory, external)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_and_amp_smoke() -> None:
    config = reduced_config("springs")
    bundle = build_datasets(config, DATA_ROOT)
    batch = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in collate_csg_batches([bundle.train[0]]).items()
    }
    model = CSGODE(config).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model.compute_all_losses(batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("CSG_RUN_SLOW_TESTS") != "1",
    reason="Set CSG_RUN_SLOW_TESTS=1 to run one-batch overfit checks",
)
@pytest.mark.parametrize("dataset", ["springs", "charged", "motion_walk", "pems08"])
def test_one_batch_overfit(dataset: str) -> None:
    config = reduced_config(dataset)
    bundle = build_datasets(config, DATA_ROOT)
    batch = collate_csg_batches([bundle.train[0]])
    model = CSGODE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    initial = None
    final = None
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        result = model.compute_all_losses(batch, sample_posterior=False)
        result["loss"].backward()
        optimizer.step()
        value = float(result["mse_lgode"].detach())
        initial = value if initial is None else initial
        final = value
    assert final is not None and initial is not None and final < initial

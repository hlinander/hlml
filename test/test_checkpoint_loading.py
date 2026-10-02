"""
Test roundtrip serialization and loading of checkpoints,
including loading from checkpoints where the config has changed.
"""
import os
import torch
import pytest
from pathlib import Path

from lib.train_dataclasses import TrainConfig, TrainEval, TrainRun, OptimizerConfig, ComputeConfig
from lib.models.dense import DenseConfig
from lib.data_registry import DataSineConfig
from lib.regression_metrics import create_regression_metrics
from lib.serialization import (
    serialize,
    SerializeConfig,
    deserialize_model,
    DeserializeConfig,
    load_model_from_checkpoint,
    load_checkpoint_data_config,
    load_checkpoint_train_run_json,
    list_checkpoint_epochs,
)
from lib.train import create_initial_state
from lib.stable_hash import stable_hash_str
import lib.compute_env as compute_env
from lib.compute_env_config import ComputeEnvironment, Paths
import lib.render_duck as duck


def make_train_run(d_hidden=100):
    loss = torch.nn.functional.mse_loss

    def mse_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=d_hidden),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        loss=mse_loss,
        optimizer=OptimizerConfig(optimizer=torch.optim.Adam, kwargs=dict()),
        batch_size=2,
    )
    train_eval = create_regression_metrics(loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=1,
        save_nth_epoch=1,
        validate_nth_epoch=1,
        project="test",
    )
    return train_run


@pytest.fixture
def checkpoint_env(tmp_path):
    """Redirect checkpoint storage to a temp directory, skip code copy."""
    original_env = compute_env._current_env
    compute_env._current_env = ComputeEnvironment(
        paths=Paths(checkpoints=tmp_path / "checkpoints")
    )
    os.environ["NOCOPY"] = "1"
    yield tmp_path
    os.environ.pop("NOCOPY", None)
    compute_env._current_env = original_env


@pytest.fixture
def checkpoint_env_with_code(tmp_path):
    """Redirect checkpoint storage to a temp directory, include code copy."""
    original_env = compute_env._current_env
    compute_env._current_env = ComputeEnvironment(
        paths=Paths(checkpoints=tmp_path / "checkpoints")
    )
    os.environ.pop("NOCOPY", None)
    yield tmp_path
    compute_env._current_env = original_env


def _train_and_serialize(train_run, device="cpu"):
    """Create initial state, do one forward/backward pass, and serialize."""
    state = create_initial_state(train_run, None, device)

    # Do a training step so the weights differ from init
    batch = next(iter(state.train_dataloader))
    outputs = state.model(batch)
    loss = train_run.train_config.loss(outputs, batch)
    loss.backward()
    state.optimizer.step()
    state.epoch = 1

    serialize(SerializeConfig(train_run=train_run, train_epoch_state=state))
    return state


def test_roundtrip_same_config(checkpoint_env):
    """Serialize a checkpoint and load it back with the same config."""
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    deser = deserialize_model(DeserializeConfig(train_run, "cpu"), latest_ok=True)
    assert deser is not None
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"


def test_load_from_checkpoint_same_config(checkpoint_env):
    """load_model_from_checkpoint works when config hasn't changed."""
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    checkpoint_hash = stable_hash_str(train_run.train_config)
    deser = load_model_from_checkpoint(checkpoint_hash, "cpu")
    assert deser is not None
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"


def test_load_from_checkpoint_changed_config(checkpoint_env):
    """Config changed (different d_hidden), but we can still load the old
    checkpoint by hash because load_model_from_checkpoint reconstructs the
    model config from the saved train_run.json."""
    # Train and serialize with d_hidden=50
    train_run_v1 = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run_v1)
    checkpoint_hash = stable_hash_str(train_run_v1.train_config)

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    # "Change the config" — now we'd use d_hidden=80.
    # The normal deserialize_model path would fail because the hash differs
    # and the architecture wouldn't match.
    train_run_v2 = make_train_run(d_hidden=80)
    new_hash = stable_hash_str(train_run_v2.train_config)
    assert new_hash != checkpoint_hash, "Hashes should differ"

    # Normal path: can't find checkpoint for the new config
    deser_normal = deserialize_model(
        DeserializeConfig(train_run_v2, "cpu"), latest_ok=True
    )
    assert deser_normal is None, "Should not find checkpoint for changed config"

    # New path: load by old hash, model is reconstructed from saved JSON
    deser = load_model_from_checkpoint(checkpoint_hash, "cpu")
    assert deser is not None

    # Verify it loaded the d_hidden=50 model with correct weights
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"

    # Verify the loaded model actually has the old architecture (d_hidden=50)
    # Dense model has l1 (input -> d_hidden) and l2 (d_hidden -> output)
    l1_weight = deser.model.state_dict()["l1.weight"]
    assert l1_weight.shape[0] == 50


def test_load_checkpoint_data_config(checkpoint_env):
    """load_checkpoint_data_config reconstructs the data config from JSON."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    data_config = load_checkpoint_data_config(checkpoint_hash)
    assert type(data_config).__name__ == "DataSineConfig"
    # torch.Size is serialized as a list in JSON
    assert list(data_config.input_shape) == [1]
    assert list(data_config.output_shape) == [1]


def test_load_checkpoint_train_run_json(checkpoint_env):
    """load_checkpoint_train_run_json returns the saved JSON with expected structure."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    saved = load_checkpoint_train_run_json(checkpoint_hash)
    assert "__class__" in saved
    assert saved["__class__"] == "TrainRun"
    assert "__data__" in saved
    assert "train_config" in saved["__data__"]
    train_config = saved["__data__"]["train_config"]
    assert train_config["__data__"]["model_config"]["__class__"] == "DenseConfig"


def test_checkpoint_readers_accept_an_explicit_root(checkpoint_env):
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)
    checkpoint_root = checkpoint_env / "checkpoints"
    checkpoint_path = checkpoint_root / f"checkpoint_{checkpoint_hash}"
    torch.save(state.model.state_dict(), checkpoint_path / "model_epoch_0001")

    loaded = load_model_from_checkpoint(
        checkpoint_hash,
        "cpu",
        epoch=1,
        checkpoint_root=checkpoint_root,
    )
    assert loaded is not None
    assert loaded.epoch == 1
    assert type(
        load_checkpoint_data_config(
            checkpoint_hash, checkpoint_root=checkpoint_root
        )
    ).__name__ == "DataSineConfig"
    assert load_checkpoint_train_run_json(
        checkpoint_hash, checkpoint_root=checkpoint_root
    )["__class__"] == "TrainRun"
    assert list_checkpoint_epochs(
        checkpoint_hash, checkpoint_root=checkpoint_root
    ) == [1]


def test_list_checkpoint_epochs(checkpoint_env):
    """list_checkpoint_epochs finds epoch-specific checkpoint files."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    # No epoch checkpoints saved (keep_epoch_checkpoints=False by default)
    assert list_checkpoint_epochs(checkpoint_hash) == []

    # Create fake epoch checkpoint files
    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"
    for epoch in [0, 20, 40]:
        (checkpoint_path / f"model_epoch_{epoch:04d}").touch()

    epochs = list_checkpoint_epochs(checkpoint_hash)
    assert epochs == [0, 20, 40]


def test_load_epoch_checkpoint(checkpoint_env):
    """load_model_from_checkpoint with epoch parameter loads the right file."""
    train_run = make_train_run(d_hidden=50)
    state = _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    # Save an epoch-specific checkpoint
    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"
    torch.save(state.model.state_dict(), checkpoint_path / "model_epoch_0001")

    original_params = {
        k: v.clone() for k, v in state.model.state_dict().items()
    }

    deser = load_model_from_checkpoint(checkpoint_hash, "cpu", epoch=1)
    assert deser is not None
    assert deser.epoch == 1
    for k, v in deser.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Mismatch on {k}"


def test_load_missing_epoch_returns_none(checkpoint_env):
    """load_model_from_checkpoint returns None for a non-existent epoch."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    deser = load_model_from_checkpoint(checkpoint_hash, "cpu", epoch=999)
    assert deser is None


def test_setup_duck_from_checkpoint(checkpoint_env):
    """setup_duck_from_checkpoint records model parameters from saved JSON."""
    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    # Clean duck state
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    saved_json = load_checkpoint_train_run_json(checkpoint_hash)
    run_id = duck.setup_duck_from_checkpoint(checkpoint_hash, 42, saved_json)

    # Verify model was inserted
    result = duck.CONN.execute(
        "SELECT train_id FROM models WHERE id = 42"
    ).fetchone()
    assert result is not None
    assert result[0] == checkpoint_hash

    # Verify parameters were recorded
    params = duck.CONN.execute(
        "SELECT name FROM model_parameter WHERE model_id = 42 AND name = 'train_config_hash'"
    ).fetchone()
    assert params is not None

    # Cleanup
    duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False


def test_code_saved_in_checkpoint(checkpoint_env_with_code):
    """archive_code_for_run saves a tar.gz code snapshot into the checkpoint directory."""
    from lib.files import archive_code_for_run

    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"

    archive_code_for_run(checkpoint_path, train_run.run_id)

    import tarfile
    tar_path = checkpoint_path / f"code_run_{train_run.run_id}.tar.gz"
    assert tar_path.is_file(), "code tar.gz should exist in checkpoint"
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
    assert any(n.endswith(".py") for n in names), "archive should contain Python files"
    assert "lib/serialization.py" in names
    assert "lib/train_dataclasses.py" in names


def test_code_saved_only_once(checkpoint_env_with_code):
    """archive_code_for_run skips if the archive already exists."""
    from lib.files import archive_code_for_run

    train_run = make_train_run(d_hidden=50)
    _train_and_serialize(train_run)
    checkpoint_hash = stable_hash_str(train_run.train_config)

    from lib.compute_env import env
    checkpoint_path = env().paths.checkpoints / f"checkpoint_{checkpoint_hash}"

    archive_code_for_run(checkpoint_path, train_run.run_id)
    tar_path = checkpoint_path / f"code_run_{train_run.run_id}.tar.gz"
    mtime_first = tar_path.stat().st_mtime

    # Call again — should be a no-op
    archive_code_for_run(checkpoint_path, train_run.run_id)
    mtime_second = tar_path.stat().st_mtime
    assert mtime_first == mtime_second, "archive should not be overwritten on second call"


def _make_grokking_train_run(train_size=64, epochs=10):
    """Create a small grokking-style config for testing resume behavior."""
    from lib.models.grok_mlp import GrokMLPConfig
    from lib.datasets.sparse_parity import DataSparseParityConfig
    from lib.classification_metrics import create_classification_metrics

    ce = torch.nn.CrossEntropyLoss()

    def ce_loss(output, batch):
        return ce(output["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=GrokMLPConfig(width=32, depth=2, activation="relu"),
        train_data_config=DataSparseParityConfig(
            d=10, k=2, n_samples=train_size, seed=0, subset_seed=0
        ),
        val_data_config=DataSparseParityConfig(
            d=10, k=2, n_samples=train_size, seed=1, subset_seed=0
        ),
        loss=ce_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(lr=1e-3, weight_decay=1.0),
        ),
        batch_size=train_size,
        ensemble_id=0,
    )
    train_eval = create_classification_metrics(None, 2)
    return TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epochs,
        save_nth_epoch=1,
        validate_nth_epoch=epochs,
        project="test_resume",
    )


def _run_n_steps(state, train_run, n, device="cpu"):
    """Run n full-batch training steps, return loss after each."""
    from lib.train_dataclasses import TrainEpochSpec

    losses = []
    for _ in range(n):
        state.model.train()
        batch = next(iter(state.train_dataloader))
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        output = state.model(batch)
        loss = train_run.train_config.loss(output, batch)
        loss.backward()
        state.optimizer.step()
        state.optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        state.epoch += 1
        state.batch += 1
    return losses


def test_resume_model_params_match(checkpoint_env):
    """Model parameters are identical after serialize/deserialize roundtrip."""
    from lib.serialization import deserialize

    train_run = _make_grokking_train_run()
    state = create_initial_state(train_run, None, "cpu")

    # Train 20 steps to warm up optimizer (Adam m/v buffers need many steps)
    _run_n_steps(state, train_run, 20)

    # Serialize
    serialize(SerializeConfig(train_run=train_run, train_epoch_state=state))

    original_params = {k: v.clone() for k, v in state.model.state_dict().items()}

    # Deserialize
    config = DeserializeConfig(train_run=train_run, device_id="cpu")
    restored = deserialize(config)
    assert restored is not None

    for k, v in restored.model.state_dict().items():
        assert torch.equal(v, original_params[k]), f"Model param mismatch on {k}"


def test_resume_optimizer_state_match(checkpoint_env):
    """Optimizer state (Adam m/v buffers, step count) is identical after roundtrip."""
    from lib.serialization import deserialize

    train_run = _make_grokking_train_run()
    state = create_initial_state(train_run, None, "cpu")

    _run_n_steps(state, train_run, 20)
    serialize(SerializeConfig(train_run=train_run, train_epoch_state=state))

    original_opt_sd = state.optimizer.state_dict()

    config = DeserializeConfig(train_run=train_run, device_id="cpu")
    restored = deserialize(config)
    restored_opt_sd = restored.optimizer.state_dict()

    # Compare state dicts (which use canonical integer keys)
    assert original_opt_sd["state"].keys() == restored_opt_sd["state"].keys()
    for k in original_opt_sd["state"]:
        for sk in original_opt_sd["state"][k]:
            orig = original_opt_sd["state"][k][sk]
            rest = restored_opt_sd["state"][k][sk]
            if torch.is_tensor(orig):
                assert torch.equal(orig, rest), f"Optimizer state mismatch: param {k}, key {sk}"
            else:
                assert orig == rest, f"Optimizer state mismatch: param {k}, key {sk}: {orig} vs {rest}"


def test_resume_preserves_ensemble_dataloader_seed(checkpoint_env, monkeypatch):
    import lib.serialization as serialization

    train_run = _make_grokking_train_run()
    train_run.train_config.ensemble_id = 3
    state = create_initial_state(train_run, None, "cpu")
    serialize(SerializeConfig(train_run=train_run, train_epoch_state=state))

    observed_seeds = []
    original = serialization.make_dataloader

    def record_seed(ds, run, device_id, shuffle, seed):
        observed_seeds.append(seed)
        return original(ds, run, device_id, shuffle, seed)

    monkeypatch.setattr(serialization, "make_dataloader", record_seed)
    restored = serialization.deserialize(
        DeserializeConfig(train_run=train_run, device_id="cpu")
    )

    assert restored is not None
    assert observed_seeds == [3, 3]


def test_resume_produces_identical_next_step(checkpoint_env):
    """The first training step after resume produces the same loss as continuous training."""
    from lib.serialization import deserialize

    train_run = _make_grokking_train_run()

    # Path A: train 20 steps, then 3 more continuously
    torch.manual_seed(42)
    state_a = create_initial_state(train_run, None, "cpu")
    _run_n_steps(state_a, train_run, 20)
    losses_a = _run_n_steps(state_a, train_run, 3)

    # Path B: train 20 steps, serialize, deserialize, then 3 more
    torch.manual_seed(42)
    state_b = create_initial_state(train_run, None, "cpu")
    _run_n_steps(state_b, train_run, 20)
    serialize(SerializeConfig(train_run=train_run, train_epoch_state=state_b))
    config = DeserializeConfig(train_run=train_run, device_id="cpu")
    state_b = deserialize(config)
    losses_b = _run_n_steps(state_b, train_run, 3)

    for i, (la, lb) in enumerate(zip(losses_a, losses_b)):
        assert abs(la - lb) < 1e-6, f"Step {i} loss diverged: continuous={la}, resumed={lb}"

    # Also verify params match
    for k in state_a.model.state_dict():
        assert torch.allclose(
            state_a.model.state_dict()[k],
            state_b.model.state_dict()[k],
            atol=1e-6,
        ), f"Params diverged on {k} after continued training"


def test_ingest_parquets_no_duplicate_export(checkpoint_env):
    """After ingesting parquets on resume, the exporter should not re-export the same rows."""
    from lib.staging_filesystem import flush_table_to_filesystem

    train_run = _make_grokking_train_run()
    state = create_initial_state(train_run, None, "cpu")

    # Train a few steps and insert metrics into duck
    for step in range(5):
        state.model.train()
        batch = next(iter(state.train_dataloader))
        output = state.model(batch)
        loss_val = train_run.train_config.loss(output, batch)
        loss_val.backward()
        state.optimizer.step()
        state.optimizer.zero_grad(set_to_none=True)
        state.batch += 1
        state.epoch += 1
        duck.insert_train_step_metric(
            state.model_id, train_run.run_id, "accuracy", state.batch, loss_val.item()
        )

    # Flush metrics to checkpoint analytics dir
    from lib.paths import get_or_create_checkpoint_path
    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    analytics_dir = checkpoint_path / "analytics"
    cursor = duck.CONN.cursor()
    flush_table_to_filesystem(
        "train_step_metric", analytics_dir, cursor, run_id=train_run.run_id,
    )

    # Verify parquets were written
    parquet_files_before = list((analytics_dir / "train_step_metric").glob("*.parquet"))
    assert len(parquet_files_before) > 0

    # Count rows exported
    import duckdb
    first_export_rows = duckdb.execute(
        f"SELECT COUNT(*) FROM read_parquet('{analytics_dir}/train_step_metric/*.parquet')"
    ).fetchone()[0]
    assert first_export_rows == 5

    # Simulate resume: reset duck, re-ensure schema, ingest parquets
    duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False
    duck.ensure_duck(train_run)
    duck.insert_model_with_model_id(train_run, state.model_id)
    duck.ingest_checkpoint_parquets(checkpoint_path, up_to_step=state.batch)

    # Verify ingested data is in memory
    in_memory_count = duck.execute_and_fetch(
        "SELECT COUNT(*) FROM train_step_metric"
    )[0][0]
    assert in_memory_count == 5

    # Now try to flush again - should export nothing (sync timestamps advanced)
    cursor = duck.CONN.cursor()
    result = flush_table_to_filesystem(
        "train_step_metric", analytics_dir, cursor, run_id=train_run.run_id,
    )
    assert result is None, "Exporter should not re-export ingested rows"

    # Train a few more steps and insert new metrics
    for step in range(3):
        state.model.train()
        batch = next(iter(state.train_dataloader))
        output = state.model(batch)
        loss_val = train_run.train_config.loss(output, batch)
        loss_val.backward()
        state.optimizer.step()
        state.optimizer.zero_grad(set_to_none=True)
        state.batch += 1
        state.epoch += 1
        duck.insert_train_step_metric(
            state.model_id, train_run.run_id, "accuracy", state.batch, loss_val.item()
        )

    # Flush again - should only export the 3 new rows
    result = flush_table_to_filesystem(
        "train_step_metric", analytics_dir, cursor, run_id=train_run.run_id,
    )
    assert result is not None, "Exporter should export new rows"

    # Count total rows across all parquet files
    total_rows = duckdb.execute(
        f"SELECT COUNT(*) FROM read_parquet('{analytics_dir}/train_step_metric/*.parquet')"
    ).fetchone()[0]
    assert total_rows == 8, f"Expected 5 + 3 = 8 total rows, got {total_rows}"

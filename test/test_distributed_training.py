"""
Test distributed training request/lock system with concurrent workers.
"""
import sys
import threading
import time
import pytest
from pathlib import Path
from types import SimpleNamespace

from lib.train_distributed import (
    request_train_run,
    fetch_requested_train_run,
    report_done,
    get_request_path_from_hash,
)
from lib.train import (
    _validate_required_resume_state,
    do_training,
    load_or_create_state,
)
from lib.stable_hash import stable_hash_str
import lib.render_duck as duck

sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_train_run


def test_required_checkpoint_resume_never_falls_back_to_fresh_state(
    tmp_path, monkeypatch
):
    train_run = create_train_run()
    train_run.notes = {
        "require_checkpoint_resume": True,
        "minimum_resume_epoch": 1,
    }
    monkeypatch.setattr("lib.train.prepare_results", lambda *_args: tmp_path)
    monkeypatch.setattr("lib.train.deserialize", lambda _config: None)

    def unexpected_fresh_state(*_args, **_kwargs):
        raise AssertionError("strict continuation attempted fresh training")

    monkeypatch.setattr("lib.train.create_initial_state", unexpected_fresh_state)

    with pytest.raises(RuntimeError, match="required checkpoint resume"):
        load_or_create_state(train_run, "cpu")


def test_required_checkpoint_resume_reconciles_optimizer_and_batch_counters():
    train_run = create_train_run()
    train_run.notes = {
        "require_checkpoint_resume": True,
        "minimum_resume_epoch": 1,
        "optimizer_updates_per_epoch": 2,
        "serialized_batch_increments_per_epoch": 4,
    }
    state = SimpleNamespace(
        epoch=1,
        batch=4,
        optimizer=SimpleNamespace(state={0: {"step": 2}, 1: {"step": 2}}),
    )

    _validate_required_resume_state(train_run, state, 1)
    state.batch = 3
    with pytest.raises(RuntimeError, match="batch counter"):
        _validate_required_resume_state(train_run, state, 1)


def test_segment_resume_preserves_inherited_counters():
    notes = dict(optimizer_updates_per_epoch=503,
                 serialized_batch_increments_per_epoch=1006,
                 resume_accounting_base_epoch=500,
                 resume_accounting_base_optimizer_step=251500,
                 resume_accounting_base_serialized_step=251500)
    run = SimpleNamespace(notes=notes)
    state = SimpleNamespace(epoch=600, batch=352100,
                            optimizer=SimpleNamespace(state={0: {'step': 301800}}))
    _validate_required_resume_state(run, state, 500)
    state.batch += 1
    with pytest.raises(RuntimeError, match='batch counter'):
        _validate_required_resume_state(run, state, 500)


@pytest.mark.parametrize('base', [-1, 601])
def test_segment_resume_rejects_invalid_base(base):
    run = SimpleNamespace(notes=dict(optimizer_updates_per_epoch=1,
                          serialized_batch_increments_per_epoch=1,
                          resume_accounting_base_epoch=base))
    state = SimpleNamespace(epoch=600, batch=600,
                            optimizer=SimpleNamespace(state={0: {'step': 600}}))
    with pytest.raises(RuntimeError, match='accounting base'):
        _validate_required_resume_state(run, state, 0)


@pytest.fixture
def distributed_env(tmp_path, monkeypatch):
    """Set up isolated distributed training environment."""
    request_dir = tmp_path / "distributed_training_requests"
    request_dir.mkdir()
    monkeypatch.setattr(
        "lib.train_distributed.DISTRIBUTED_TRAINING_REQUEST_PATH", request_dir
    )

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    monkeypatch.setattr(
        "lib.paths.get_lock_path",
        lambda tc, lock_name="": lock_dir / f"lock{lock_name}_{stable_hash_str(tc)}",
    )

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    from lib.compute_env_config import Paths
    mock_paths = Paths(
        checkpoints=checkpoint_dir,
        locks=lock_dir,
        distributed_requests=request_dir,
        artifacts=tmp_path / "artifacts",
        datasets=tmp_path / "datasets",
    )

    from lib.compute_env_config import ComputeEnvironment
    mock_env = ComputeEnvironment(paths=mock_paths)
    monkeypatch.setattr("lib.compute_env._current_env", mock_env)

    # Set up analytics config
    from lib.analytics_config import AnalyticsConfig, StagingFilesystem, CentralDuckDB
    import lib.analytics_config as analytics_config_module

    analytics_config_module._analytics_config = AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=tmp_path / "staging",
            archive_dir=tmp_path / "archive",
        ),
        central=CentralDuckDB(db_path=tmp_path / "central.db"),
        export_interval_seconds=9999,
        ingest_interval_seconds=9999,
    )

    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    yield tmp_path

    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False
    analytics_config_module._analytics_config = None


def test_request_and_fetch(distributed_env):
    """Test that a requested train run can be fetched and locked."""
    train_run = create_train_run()

    request_train_run(train_run)

    # Verify dill file was created
    dill_files = list(
        (distributed_env / "distributed_training_requests").glob("*.dill")
    )
    assert len(dill_files) == 1

    # Fetch the request
    result = fetch_requested_train_run([train_run])
    assert result is not None
    assert result.hash == dill_files[0].stem

    # While locked, another fetch should fail
    result2 = fetch_requested_train_run([train_run])
    assert result2 is None

    # Release and report done
    report_done(result)

    # Dill file should be cleaned up
    dill_files = list(
        (distributed_env / "distributed_training_requests").glob("*.dill")
    )
    assert len(dill_files) == 0


def test_concurrent_workers_different_configs(distributed_env):
    """Test that two workers can train different configs concurrently."""
    train_run_a = create_train_run()
    train_run_b = create_train_run()
    # Ensure different configs by changing ensemble_id
    train_run_b.train_config.ensemble_id = 1

    request_train_run(train_run_a)
    request_train_run(train_run_b)

    results = {}
    errors = []

    def worker(name, train_runs):
        try:
            result = fetch_requested_train_run(train_runs)
            if result is not None:
                results[name] = result.hash
                time.sleep(0.1)  # Simulate work
                report_done(result)
        except Exception as e:
            errors.append((name, e))

    t1 = threading.Thread(target=worker, args=("w1", [train_run_a]))
    t2 = threading.Thread(target=worker, args=("w2", [train_run_b]))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Worker errors: {errors}"
    assert len(results) == 2, "Both workers should have acquired a config"
    assert results["w1"] != results["w2"], "Workers should train different configs"


def test_concurrent_workers_same_config(distributed_env):
    """Test that only one worker can train a given config at a time."""
    train_run = create_train_run()
    request_train_run(train_run)

    acquired = []
    failed = []

    def worker(name):
        result = fetch_requested_train_run([train_run])
        if result is not None:
            acquired.append(name)
            time.sleep(0.2)  # Hold the lock
            report_done(result)
        else:
            failed.append(name)

    t1 = threading.Thread(target=worker, args=("w1",))
    t2 = threading.Thread(target=worker, args=("w2",))
    t1.start()
    time.sleep(0.05)  # Ensure w1 acquires first
    t2.start()
    t1.join()
    t2.join()

    assert len(acquired) == 1, "Only one worker should acquire the config"
    assert len(failed) == 1, "The other worker should fail to acquire"


def test_checkpoint_lock_prevents_concurrent_training(distributed_env):
    """Test that the checkpoint lock prevents two workers from training the same config."""
    train_run = create_train_run()
    train_run.epochs = 1

    duck.ensure_duck(None, True)

    lock_acquired_order = []
    lock = threading.Lock()

    def worker(name):
        state = load_or_create_state(train_run, "cpu")
        try:
            do_training(train_run, state, "cpu")
            with lock:
                lock_acquired_order.append((name, "completed"))
        except Exception as e:
            with lock:
                lock_acquired_order.append((name, f"failed: {type(e).__name__}"))

    # Run two workers trying to train the same config
    t1 = threading.Thread(target=worker, args=("w1",))
    t2 = threading.Thread(target=worker, args=("w2",))
    t1.start()
    time.sleep(0.05)
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    # One should complete, the other should fail with Timeout
    assert len(lock_acquired_order) == 2
    outcomes = [r[1] for r in lock_acquired_order]
    assert "completed" in outcomes, f"At least one worker should complete: {lock_acquired_order}"
    assert any("Timeout" in o for o in outcomes), \
        f"One worker should timeout on checkpoint lock: {lock_acquired_order}"


def test_report_done_uses_stored_hash(distributed_env):
    """Test that report_done uses the stored hash, not re-hashed train_run."""
    from lib.train_distributed import DistributedTrainRun
    from filelock import FileLock

    # Create a fake dill file with a known hash
    fake_hash = "abcdef0123456789"
    dill_path = get_request_path_from_hash(fake_hash)
    dill_path.write_bytes(b"fake")

    lock_path = distributed_env / "distributed_training_requests" / f"lock_{fake_hash}"
    lock = FileLock(str(lock_path))
    lock.acquire()

    train_run = create_train_run()
    result = DistributedTrainRun(train_run=train_run, hash=fake_hash, lock=lock)

    assert dill_path.exists()
    report_done(result)
    assert not dill_path.exists(), "Dill file should be deleted using stored hash"

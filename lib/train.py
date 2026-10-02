import sys
import traceback
from pathlib import Path
import torch
import logging
import time
from typing import Dict
from typing import Optional

from filelock import FileLock, Timeout
from lib.metric import MetricSample, Metric
from lib.timing_metric import Timing
import lib.data_factory as data_factory
from lib.data_utils import get_sampler, make_dataloader
import lib.model_factory as model_factory
import lib.render_duck as duck

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainEpochSpec
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import ComputeConfig

from lib.serialization import SerializeConfig
from lib.serialization import DeserializeConfig
from lib.serialization import deserialize
from lib.serialization import get_serialized_epoch
from lib.serialization import serialize

from lib.train_visualization import visualize_progress_batches, visualize_progress

from lib.export import export_all
from lib.paths import get_lock_path, get_or_create_checkpoint_path
from lib.files import prepare_results, archive_code_for_run
import lib.ddp as ddp


def evaluate_metrics_on_data(
    model,
    model_id,
    step,
    metrics: Dict[str, Metric],
    data_config,
    batch_size: int,
    compute_config: ComputeConfig,
    device,
):
    dataloader = create_dataloader(
        data_config, batch_size, shuffle=False, compute_config=compute_config
    )
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            output = model(batch)

            metric_sample = MetricSample(
                output=output,
                batch=batch,
                model_id=model_id,
            )
            for metric in metrics:
                values = metric.per_sample(metric_sample)

                if values.dtype == torch.double:
                    values = values.float()

                if (
                    len(values.shape) == 1
                    and values.shape[0] == batch["sample_id"].shape[0]
                ):
                    value_per_sample = values.tolist()
                else:
                    value_per_sample = None

                duck.insert_checkpoint_sample_metric(
                    metric_sample.model_id,
                    step,
                    metric.name(),
                    data_factory.get_factory().get_class(data_config).__name__,
                    batch["sample_id"].tolist(),
                    values.mean().item(),
                    value_per_sample,
                )


def flush_epoch_metrics(
    train_epoch_state: TrainEpochState,
    train_run: TrainRun,
):
    """Flush accumulated epoch-level metrics to duck and reset accumulators."""
    train_dataset = (
        data_factory.get_factory()
        .get_class(train_run.train_config.train_data_config)
        .__name__
    )

    # Training metrics
    for metric in train_epoch_state.train_metrics:
        result = metric.epoch_accumulator.result()
        if result is not None:
            mean, min_val, max_val, count = result
            duck.insert_train_epoch_metric(
                train_epoch_state.model_id,
                train_run.run_id,
                train_epoch_state.epoch,
                train_epoch_state.batch,
                metric.name(),
                train_dataset,
                "train",
                mean, min_val, max_val, count,
            )
        metric.epoch_accumulator.reset()

    # Built-in training diagnostics
    for name, acc in [
        ("gradient_norm", train_epoch_state.gradient_norm_accumulator),
        ("grad_clip_ratio", train_epoch_state.grad_clip_ratio_accumulator),
        ("learning_rate", train_epoch_state.learning_rate_accumulator),
        ("parameter_norm", train_epoch_state.parameter_norm_accumulator),
        ("batch_time", train_epoch_state.batch_time_accumulator),
        ("batch_time_with_diagnostics", train_epoch_state.batch_time_diag_accumulator),
    ]:
        result = acc.result()
        if result is not None:
            mean, min_val, max_val, count = result
            duck.insert_train_epoch_metric(
                train_epoch_state.model_id,
                train_run.run_id,
                train_epoch_state.epoch,
                train_epoch_state.batch,
                name,
                train_dataset,
                "train",
                mean, min_val, max_val, count,
            )
        acc.reset()

    # Validation metrics
    if train_run.train_config.val_data_config is not None:
        val_dataset = (
            data_factory.get_factory()
            .get_class(train_run.train_config.val_data_config)
            .__name__
        )
        for metric in train_epoch_state.validation_metrics:
            result = metric.epoch_accumulator.result()
            if result is not None:
                mean, min_val, max_val, count = result
                duck.insert_train_epoch_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    train_epoch_state.epoch,
                    train_epoch_state.batch,
                    metric.name(),
                    val_dataset,
                    "val",
                    mean, min_val, max_val, count,
                )
            metric.epoch_accumulator.reset()


def validate(
    train_epoch_state: TrainEpochState,
    train_epoch_spec: TrainEpochSpec,
    train_run: TrainRun,
):
    if train_epoch_state.val_dataloader is None:
        return
    if train_epoch_state.epoch % train_run.validate_nth_epoch != 0:
        # Evaluate if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            return
    model = train_epoch_state.model
    dataloader = train_epoch_state.val_dataloader
    device = train_epoch_spec.device_id

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
            output = model(batch)
            if ddp.get_rank() == 0:
                metric_sample = MetricSample(
                    output=output,
                    batch=batch,
                    model_id=train_epoch_state.model_id,
                )
                for metric in train_epoch_state.validation_metrics:
                    values = metric.per_sample(metric_sample)

                    if values.dtype == torch.double:
                        values = values.float()

                    if (
                        len(values.shape) == 1
                        and values.shape[0] == batch["sample_id"].shape[0]
                    ):
                        value_per_sample = values.tolist()
                    else:
                        value_per_sample = None

                    metric.epoch_accumulator.update(values.mean().item())

                    duck.insert_checkpoint_sample_metric(
                        metric_sample.model_id,
                        train_epoch_state.batch,
                        metric.name(),
                        data_factory.get_factory()
                        .get_class(train_run.train_config.val_data_config)
                        .__name__,
                        batch["sample_id"].tolist(),
                        values.mean().item(),
                        value_per_sample,
                    )

    if train_run.post_validate_hook is not None and ddp.get_rank() == 0:
        train_run.post_validate_hook(model, train_run, train_epoch_state, device)


def train(
    train_run: TrainRun,
    train_epoch_state: TrainEpochState,
    train_epoch_spec: TrainEpochSpec,
):
    model = train_epoch_state.model
    dataloader = train_epoch_state.train_dataloader
    loss = train_epoch_spec.loss
    optimizer = train_epoch_state.optimizer
    device = train_epoch_spec.device_id

    model.train()

    if train_run.train_config.model_pre_train_hook is not None:
        model = train_run.train_config.model_pre_train_hook(model, train_run=train_run)

    if dataloader.sampler.__class__.__name__ == "DistributedSampler":
        dataloader.sampler.set_epoch(train_epoch_state.epoch)

    diagnostics_interval = train_run.train_eval.diagnostics_interval
    visualizers = [visualize_progress_batches, visualize_progress]
    for i, batch in enumerate(dataloader):
        batch_time = train_epoch_state.timing_metric.stop("batch")
        if batch_time is not None:
            if train_epoch_state._last_step_had_diagnostics:
                batch_time_name = "batch_time_with_diagnostics"
                train_epoch_state.batch_time_diag_accumulator.update(batch_time)
            else:
                batch_time_name = "batch_time"
                train_epoch_state.batch_time_accumulator.update(batch_time)
            if ddp.get_rank() == 0:
                duck.insert_train_step_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    batch_time_name,
                    train_epoch_state.batch,
                    batch_time,
                )
        train_epoch_state.timing_metric.start("batch")
        train_epoch_state.batch += ddp.get_world_size()

        run_diagnostics = (i % diagnostics_interval == 0)
        train_epoch_state._last_step_had_diagnostics = run_diagnostics

        batch = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
        output = model(batch)

        loss_val = loss(output, batch)
        loss_val.backward()

        t_db = time.time()
        if ddp.get_rank() == 0 and train_run.train_eval.log_sample_ids:
            t_train_step = time.time()
            duck.insert_train_step(
                train_epoch_state.model_id,
                train_run.run_id,
                train_epoch_state.batch,
                data_factory.get_factory()
                .get_class(train_run.train_config.train_data_config)
                .__name__,
                batch["sample_id"].long().tolist(),
            )
            t_train_step = time.time() - t_train_step
            if run_diagnostics:
                duck.insert_train_step_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    "t_insert_train_step",
                    train_epoch_state.batch,
                    t_train_step,
                )

        # Gradient clipping always runs (when configured), independent of diagnostics
        if train_run.train_config.gradient_clipping is not None:
            max_norm = train_run.train_config.gradient_clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm
            ).item()

        if run_diagnostics:
            # Gradient norm (reuse clip_grad_norm_ result if available)
            if train_run.train_eval.log_gradient_norm:
                if train_run.train_config.gradient_clipping is None:
                    grad_norm = sum(
                        p.grad.detach().norm().pow(2)
                        for p in model.parameters()
                        if p.grad is not None
                    ).sqrt().item()
                train_epoch_state.gradient_norm_accumulator.update(grad_norm)
                if ddp.get_rank() == 0:
                    duck.insert_train_step_metric(
                        train_epoch_state.model_id,
                        train_run.run_id,
                        "gradient_norm",
                        train_epoch_state.batch,
                        grad_norm,
                    )

                # Clip ratio (only when clipping is active)
                if train_run.train_config.gradient_clipping is not None:
                    clip_ratio = min(max_norm / (grad_norm + 1e-8), 1.0)
                    train_epoch_state.grad_clip_ratio_accumulator.update(clip_ratio)
                    if ddp.get_rank() == 0:
                        duck.insert_train_step_metric(
                            train_epoch_state.model_id,
                            train_run.run_id,
                            "grad_clip_ratio",
                            train_epoch_state.batch,
                            clip_ratio,
                        )

        optimizer.step()

        if run_diagnostics:
            # Learning rate (essentially free)
            lr = optimizer.param_groups[0]['lr']
            train_epoch_state.learning_rate_accumulator.update(lr)
            if ddp.get_rank() == 0:
                duck.insert_train_step_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    "learning_rate",
                    train_epoch_state.batch,
                    lr,
                )

            # Parameter norm (O(1) extra memory)
            if train_run.train_eval.log_parameter_norm:
                param_norm = sum(
                    p.detach().norm().pow(2) for p in model.parameters()
                ).sqrt().item()
                train_epoch_state.parameter_norm_accumulator.update(param_norm)
                train_epoch_state._peak_parameter_norm = max(
                    train_epoch_state._peak_parameter_norm, param_norm
                )
                norm_ratio = param_norm / train_epoch_state._peak_parameter_norm
                if ddp.get_rank() == 0:
                    duck.insert_train_step_metric(
                        train_epoch_state.model_id,
                        train_run.run_id,
                        "parameter_norm",
                        train_epoch_state.batch,
                        param_norm,
                    )
                    duck.insert_train_step_metric(
                        train_epoch_state.model_id,
                        train_run.run_id,
                        "norm_ratio",
                        train_epoch_state.batch,
                        norm_ratio,
                    )

        optimizer.zero_grad(set_to_none=True)

        if ddp.get_rank() > 0:
            continue

        if i == 0 and torch.cuda.is_available():
            train_epoch_state.device_memory_stats = (
                torch.cuda.memory_stats_as_nested_dict()
            )

        train_epoch_state.timing_metric.start("train_metrics")
        metric_sample = MetricSample(
            output=output,
            batch=batch,
            model_id=train_epoch_state.model_id,
        )
        for metric in train_epoch_state.train_metrics:
            value = metric(metric_sample)
            metric.epoch_accumulator.update(value)
            duck.insert_train_step_metric(
                metric_sample.model_id,
                train_run.run_id,
                metric.name(),
                train_epoch_state.batch,
                value,
            )
        train_epoch_state.timing_metric.stop("train_metrics")
        t_db_elapsed = time.time() - t_db

        t_db_elapsed = time.time() - t_db

        # Log DB + metrics timing on diagnostic steps
        if run_diagnostics:
            duck.insert_train_step_metric(
                train_epoch_state.model_id,
                train_run.run_id,
                "t_db_and_metrics",
                train_epoch_state.batch,
                t_db_elapsed,
            )

        # Record GPU metrics if monitor is active
        if train_epoch_state.gpu_monitor is not None:
            train_epoch_state.gpu_monitor.record_metrics(train_epoch_state.batch)

        try:
            now = time.time()
            if now > train_epoch_state.next_visualization:
                train_epoch_state.next_visualization = (
                    time.time() + train_run.visualize_interval_s
                )
                train_epoch_state.timing_metric.start("duck")
                last_sync_result = True, ""

                duck_time = train_epoch_state.timing_metric.stop("duck")
                duck.insert_train_step_metric(
                    train_epoch_state.model_id,
                    train_run.run_id,
                    "dbsync",
                    train_epoch_state.batch,
                    duck_time,
                )
                if train_run.visualize_terminal and sys.stdout.isatty():
                    train_epoch_state.timing_metric.start("visualize")
                    visualizers[train_epoch_state.next_visualizer](
                        train_epoch_state,
                        train_run,
                        last_sync_result,
                        train_epoch_spec.device_id,
                    )
                    train_epoch_state.timing_metric.stop("visualize")
                    train_epoch_state.next_visualizer = (
                        train_epoch_state.next_visualizer + 1
                    ) % len(visualizers)
        except Exception as e:
            logging.error("Visualization failed")
            logging.error(str(e))
            print(traceback.format_exc())


def create_dataloader(
    data_config, batch_size: int, shuffle: bool, compute_config: ComputeConfig
):
    ds = data_factory.get_factory().create(data_config)
    sampler, shuffle = get_sampler(compute_config, ds, shuffle=shuffle)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        sampler=sampler,
        num_workers=compute_config.num_workers,
        collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
        pin_memory=True,
        persistent_workers=True and compute_config.num_workers > 0,
    )
    return dataloader




def create_initial_state(train_run: TrainRun, code_path: Optional[Path], device_id):
    train_config = train_run.train_config

    train_ds = data_factory.get_factory().create(train_config.train_data_config)
    train_dataloader = make_dataloader(
        train_ds, train_run, device_id, shuffle=True, seed=train_config.ensemble_id,
    )

    if train_config.val_data_config is not None:
        val_ds = data_factory.get_factory().create(train_config.val_data_config)
        val_dataloader = make_dataloader(
            val_ds, train_run, device_id, shuffle=False, seed=train_config.ensemble_id,
        )
    else:
        val_dataloader = None

    torch.manual_seed(train_config.ensemble_id)
    init_model = model_factory.get_factory().create(
        train_config.model_config,
        train_ds.__class__.data_spec(train_config.train_data_config),
    )

    if train_config.post_model_create_hook is not None:
        init_model = train_config.post_model_create_hook(
            init_model, train_run=train_run
        )

    init_model = init_model.to(device=torch.device(device_id))
    if train_run.compute_config.distributed:
        device = torch.device(device_id)
        ddp_kwargs = {"device_ids": [device_id]} if device.type == "cuda" else {}
        init_model = torch.nn.parallel.DistributedDataParallel(
            init_model,
            find_unused_parameters=getattr(
                train_run.compute_config, "find_unused_parameters", True
            ),
            **ddp_kwargs,
        )

    # Re-import classes fresh to avoid dill deserialization breaking
    # isinstance checks inside PyTorch (e.g. scheduler checking Optimizer type)
    import importlib

    opt_cls = train_config.optimizer.optimizer
    opt_module = importlib.import_module(opt_cls.__module__)
    fresh_opt_cls = getattr(opt_module, opt_cls.__name__)

    # Optimizer factories marked with takes_model receive the model itself
    # (e.g. to group parameters by module type); plain torch optimizers get
    # the usual parameter iterable.
    if getattr(fresh_opt_cls, "takes_model", False):
        opt_params = init_model
    else:
        opt_params = init_model.parameters()
    opt = fresh_opt_cls(opt_params, **train_config.optimizer.kwargs)

    scheduler = None
    if train_config.scheduler_config is not None:
        sched_cls = train_config.scheduler_config.scheduler
        sched_module = importlib.import_module(sched_cls.__module__)
        fresh_sched_cls = getattr(sched_module, sched_cls.__name__)
        scheduler = fresh_sched_cls(opt, **train_config.scheduler_config.kwargs)
    train_metrics = [metric() for metric in train_run.train_eval.train_metrics]
    validation_metrics = [
        metric() for metric in train_run.train_eval.validation_metrics
    ]

    return TrainEpochState(
        model_id=duck.insert_model(train_run) if ddp.get_rank() == 0 else 0,
        model=init_model,
        optimizer=opt,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        epoch=0,
        batch=0,
        train_dataloader=train_dataloader,
        scheduler=scheduler,
        val_dataloader=val_dataloader,
        next_visualization=0.0,
        next_visualizer=0,
        timing_metric=Timing(),
        code_path=code_path,
    )


def _validate_required_resume_state(
    train_run: TrainRun,
    state: TrainEpochState,
    minimum_resume_epoch: int | None,
) -> None:
    notes = train_run.notes
    if minimum_resume_epoch is not None and int(state.epoch) < minimum_resume_epoch:
        raise RuntimeError(
            f"checkpoint epoch {state.epoch} is below required resume "
            f"epoch {minimum_resume_epoch}"
        )
    updates_per_epoch = int(notes["optimizer_updates_per_epoch"])
    batch_increments_per_epoch = int(notes["serialized_batch_increments_per_epoch"])
    if updates_per_epoch <= 0 or batch_increments_per_epoch <= 0:
        raise RuntimeError("strict checkpoint-resume accounting is invalid")
    # A continuation may change world size while preserving the global batch.
    base_epoch = int(notes.get("resume_accounting_base_epoch", 0))
    base_optimizer = int(notes.get("resume_accounting_base_optimizer_step", 0))
    base_batch = int(notes.get("resume_accounting_base_serialized_step", 0))
    if min(base_epoch, base_optimizer, base_batch) < 0 or int(state.epoch) < base_epoch:
        raise RuntimeError("invalid strict checkpoint-resume accounting base")
    elapsed_epochs = int(state.epoch) - base_epoch
    expected_optimizer_step = base_optimizer + elapsed_epochs * updates_per_epoch
    optimizer_steps = set()
    for parameter_state in state.optimizer.state.values():
        step = parameter_state.get("step")
        if step is None:
            raise RuntimeError("strict checkpoint-resume optimizer lacks a step")
        optimizer_steps.add(int(step.item() if hasattr(step, "item") else step))
    if optimizer_steps != {expected_optimizer_step}:
        raise RuntimeError(
            f"checkpoint optimizer steps are {sorted(optimizer_steps)}, expected "
            f"{expected_optimizer_step}"
        )
    expected_batch = base_batch + elapsed_epochs * batch_increments_per_epoch
    if int(state.batch) != expected_batch:
        raise RuntimeError(
            f"checkpoint batch counter is {state.batch}, expected {expected_batch}"
        )


def load_or_create_state(train_run: TrainRun, device_id) -> TrainEpochState:
    config = DeserializeConfig(
        train_run=train_run,
        device_id=device_id,
    )
    code_path = prepare_results("code", train_run)
    state = None
    notes = train_run.notes if isinstance(train_run.notes, dict) else {}
    require_checkpoint_resume = bool(notes.get("require_checkpoint_resume", False))
    minimum_resume_epoch = notes.get("minimum_resume_epoch")
    if minimum_resume_epoch is not None:
        minimum_resume_epoch = int(minimum_resume_epoch)
        if not require_checkpoint_resume or minimum_resume_epoch < 0:
            raise RuntimeError("invalid strict checkpoint-resume requirements")
    try:
        state = deserialize(config)
        if state is not None:
            if require_checkpoint_resume:
                _validate_required_resume_state(
                    train_run, state, minimum_resume_epoch
                )
            if ddp.get_rank() == 0:
                duck.insert_model_with_model_id(train_run, state.model_id)
                checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
                duck.ingest_checkpoint_parquets(checkpoint_path, up_to_step=state.batch)
                peak = duck.select_max_train_step_metric(state.model_id, "parameter_norm")
                if peak is not None:
                    state._peak_parameter_norm = peak
            print(f"Resuming from epoch {state.epoch}/{train_run.epochs} (batch {state.batch})")
    except Exception as e:
        if require_checkpoint_resume:
            raise RuntimeError("required checkpoint resume failed") from e
        print("ERROR: Failed to load checkpoint, creating a new initial state.")

    if state is None:
        if require_checkpoint_resume:
            raise RuntimeError("required checkpoint resume returned no state")
        state = create_initial_state(
            train_run=config.train_run,
            code_path=code_path,
            device_id=config.device_id,
        )
        print(f"Starting fresh training for {train_run.epochs} epochs")

    if ddp.get_rank() == 0:
        archive_code_for_run(
            get_or_create_checkpoint_path(train_run.train_config),
            train_run.run_id,
        )

    return state


def log_run_start(train_run: TrainRun, state: TrainEpochState):
    """Log config and state at the start of a training run."""
    import json
    print("=" * 60)
    print("Train config:")
    print(json.dumps(train_run.train_config.serialize_human(), indent=2, default=str))
    print(f"Project: {train_run.project}")
    print(f"Epochs: {train_run.epochs}")
    print(f"Run ID: {train_run.run_id}, model ID: {state.model_id}")
    print("=" * 60)


def log_run_done(train_run: TrainRun, start_epoch: int, end_epoch: int):
    """Log completion of a training run."""
    trained = end_epoch - start_epoch
    print("=" * 60)
    if trained > 0:
        print(f"Done. Trained epochs {start_epoch + 1}-{end_epoch} ({trained} epochs)")
    else:
        print(f"Done. Already at epoch {end_epoch}/{train_run.epochs}, no training needed")
    print("=" * 60)


def do_training_unlocked(train_run: TrainRun, state: TrainEpochState, device_id):
    if ddp.get_rank() == 0:
        log_run_start(train_run, state)

    start_epoch = state.epoch
    train_epoch_spec = TrainEpochSpec(
        loss=train_run.train_config.loss,
        device_id=device_id,
    )

    print("Serializing config...")
    serialize_config = SerializeConfig(train_run=train_run, train_epoch_state=state)

    if ddp.get_rank() == 0:
        try:
            print("Render duck")
            duck.render_duck(train_run, state)
        except duck.duckdb.duckdb.ConstraintException:
            print("Probably already synced model parameters...")

    # Start periodic export to staging (S3 or filesystem based on config)
    if ddp.get_rank() == 0:
        from lib.export import start_periodic_export
        from lib.analytics_config import analytics_config

        config = analytics_config()
        print(
            f"Starting periodic export to {config.staging.type} staging (interval: {config.export_interval_seconds}s)..."
        )
        _export_thread = start_periodic_export(train_run)
        serialize(serialize_config)

    # Start GPU monitoring (if available)
    if ddp.get_rank() == 0:
        from lib.gpu_monitor import GPUMonitor

        state.gpu_monitor = GPUMonitor(
            state.model_id, train_run.run_id, train_run.gpu_monitor
        )
        state.gpu_monitor.start()

    print("Run epochs...")
    while state.epoch < train_run.epochs:
        train(train_run, state, train_epoch_spec)

        if state.scheduler is not None:
            state.scheduler.step()

        state.epoch += 1
        if ddp.get_rank() == 0:
            duck.insert_checkpoint(state.model_id, state.batch, None)

        validate(state, train_epoch_spec, train_run)
        if ddp.get_rank() == 0:
            flush_epoch_metrics(state, train_run)
            serialize(serialize_config)
        if train_run.compute_config.distributed:
            torch.distributed.barrier()

    if ddp.get_rank() > 0:
        return

    # Final export to ensure all metrics are flushed
    try:
        print("Final analytics export...")
        export_all(train_run)
    except Exception as e:
        logging.error(f"Final export failed: {e}")

    log_run_done(train_run, start_epoch, state.epoch)


def do_training(train_run: TrainRun, state: TrainEpochState, device_id):
    checkpoint_path = get_lock_path(train_run.train_config)
    # A caller that loaded state before another worker acquired the lock must
    # not wait and then train that stale state after the first worker exits.
    lock = FileLock(f"{checkpoint_path}", timeout=0)
    if not train_run.compute_config.distributed:
        with lock:
            latest_epoch = get_serialized_epoch(train_run.train_config)
            if latest_epoch is not None and latest_epoch > state.epoch:
                raise Timeout(f"stale training state for {checkpoint_path}")
            if state.epoch >= train_run.epochs:
                log_run_done(train_run, state.epoch, state.epoch)
                return
            do_training_unlocked(train_run, state, device_id)
        return

    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed training requested before initializing a process group; "
            "launch through run.py --mode torchrun."
        )

    training_complete = [state.epoch >= train_run.epochs]
    torch.distributed.broadcast_object_list(training_complete, src=0)
    if training_complete[0]:
        if ddp.get_rank() == 0:
            log_run_done(train_run, state.epoch, state.epoch)
        return

    lock_status = [False]
    if ddp.get_rank() == 0:
        try:
            lock.acquire()
            latest_epoch = get_serialized_epoch(train_run.train_config)
            if latest_epoch is not None and latest_epoch > state.epoch:
                lock.release()
            else:
                lock_status[0] = True
        except Timeout:
            pass
    torch.distributed.broadcast_object_list(lock_status, src=0)

    if not lock_status[0]:
        raise Timeout(str(checkpoint_path))

    try:
        do_training_unlocked(train_run, state, device_id)
        # Nonzero ranks return before rank zero's final export. Keep the
        # checkpoint lock until the entire distributed run has completed.
        torch.distributed.barrier()
    finally:
        if ddp.get_rank() == 0:
            lock.release()

#!/usr/bin/env python
"""Small end-to-end DDP smoke test for the project training framework."""

import json
import importlib
import os
from pathlib import Path

import torch

from lib.compute_env import env as compute_env
from lib.data_registry import DataSineConfig
from lib.ddp import ddp_setup, get_rank, get_world_size
from lib.distributed_trainer import distributed_train
from lib.models.dense import DenseConfig
from lib.regression_metrics import create_regression_metrics
from lib.train import do_training, load_or_create_state
from lib.train_dataclasses import (
    ComputeConfig,
    GPUMonitorConfig,
    OptimizerConfig,
    TrainConfig,
    TrainRun,
)


def configure_isolated_environment():
    root_value = os.environ.get("EP_DDP_SMOKE_ROOT")
    if root_value is None:
        return

    from lib.analytics_config import AnalyticsConfig, CentralDuckDB, StagingFilesystem
    import lib.analytics_config as analytics_config_module
    from lib.compute_env_config import ComputeEnvironment, Paths

    root = Path(root_value)
    compute_env_module = importlib.import_module("lib.compute_env")
    compute_env_module._current_env = ComputeEnvironment(
        paths=Paths(
            checkpoints=root / "checkpoints",
            locks=root / "locks",
            distributed_requests=root / "distributed_requests",
            artifacts=root / "artifacts",
            datasets=root / "datasets",
        )
    )
    analytics_config_module._analytics_config = AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=root / "staging",
            archive_dir=root / "archive",
        ),
        central=CentralDuckDB(db_path=root / "analytics.db"),
        export_interval_seconds=9999,
        ingest_interval_seconds=9999,
    )


def create_config() -> TrainRun:
    mse_loss = torch.nn.functional.mse_loss

    def loss(output, batch):
        return mse_loss(output["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=DenseConfig(d_hidden=16),
        train_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        val_data_config=DataSineConfig(
            input_shape=torch.Size([1]), output_shape=torch.Size([1])
        ),
        loss=loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam,
            kwargs={"lr": 1e-3},
        ),
        batch_size=10,
        ensemble_id=20260831,
        _version=4,
    )
    return TrainRun(
        project="ddp_smoke",
        compute_config=ComputeConfig(
            distributed=True,
            num_workers=0,
            num_gpus=2,
            find_unused_parameters=False,
        ),
        train_config=train_config,
        train_eval=create_regression_metrics(torch.nn.functional.mse_loss, None),
        epochs=1,
        save_nth_epoch=1,
        validate_nth_epoch=1,
        visualize_terminal=False,
        gpu_monitor=GPUMonitorConfig(enabled=False),
    )


def verify_parameters_match(state, device) -> float:
    model = state.model.module
    checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in model.parameters()]
    ).sum()
    checksum = checksum.to(device)
    gathered = [torch.zeros_like(checksum) for _ in range(get_world_size())]
    torch.distributed.all_gather(gathered, checksum)
    if not all(torch.allclose(gathered[0], value) for value in gathered[1:]):
        raise RuntimeError(f"DDP parameters diverged across ranks: {gathered}")
    return checksum.item()


def main():
    configure_isolated_environment()
    device = ddp_setup()
    if not torch.distributed.is_initialized():
        train_run = create_config()
        distributed_train([train_run])
        marker = Path(
            os.environ.get(
                "EP_DDP_SMOKE_MARKER",
                compute_env().paths.artifacts / "ddp_smoke_success.json",
            )
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "mode": "orchestrated",
                    "world_size": train_run.compute_config.num_gpus,
                    "epoch": train_run.epochs,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"DDP_ORCHESTRATION_SUCCESS {marker}")
        return
    if get_world_size() < 2:
        raise RuntimeError("DDP smoke test requires at least two processes")

    train_run = create_config()
    state = load_or_create_state(train_run, device)
    do_training(train_run, state, device)
    checksum = verify_parameters_match(state, device)

    if get_rank() == 0:
        marker = Path(
            os.environ.get(
                "EP_DDP_SMOKE_MARKER",
                compute_env().paths.artifacts / "ddp_smoke_success.json",
            )
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "backend": torch.distributed.get_backend(),
                    "world_size": get_world_size(),
                    "epoch": state.epoch,
                    "batch": state.batch,
                    "parameter_checksum": checksum,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"DDP_SMOKE_SUCCESS {marker}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

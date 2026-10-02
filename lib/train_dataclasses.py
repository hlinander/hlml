import sys
import os
import torch
import copy
from dataclasses import dataclass, field
from typing import List
from typing import Callable
from typing import Union
from typing import Set
from typing import Dict
from pathlib import Path

from lib.metric import Metric, EpochMetricAccumulator
from lib.timing_metric import Timing
from lib.stable_hash import stable_hash_str
from lib.git import get_rev
import lib.serialize_human
from lib.random_util import random_positive_i64


@dataclass
class TrainEpochState:
    model_id: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_metrics: List[Metric]
    validation_metrics: List[Metric]
    train_dataloader: torch.utils.data.DataLoader
    timing_metric: Timing
    epoch: int
    batch: int
    next_visualization: float
    next_visualizer: int
    scheduler: object = None
    val_dataloader: torch.utils.data.DataLoader = None
    device_memory_stats: dict = None
    host_memory_stats: object = None
    psql_query_cache: Set = field(default_factory=lambda: set())
    psql_starting_xs: Dict[str, float] = None
    code_path: Path = None
    gpu_monitor: object = None
    gradient_norm_accumulator: EpochMetricAccumulator = field(default_factory=EpochMetricAccumulator)
    parameter_norm_accumulator: EpochMetricAccumulator = field(default_factory=EpochMetricAccumulator)
    learning_rate_accumulator: EpochMetricAccumulator = field(default_factory=EpochMetricAccumulator)
    grad_clip_ratio_accumulator: EpochMetricAccumulator = field(default_factory=EpochMetricAccumulator)
    batch_time_accumulator: EpochMetricAccumulator = field(default_factory=EpochMetricAccumulator)
    batch_time_diag_accumulator: EpochMetricAccumulator = field(default_factory=EpochMetricAccumulator)
    _last_step_had_diagnostics: bool = False
    _peak_parameter_norm: float = 0.0


@dataclass
class TrainEpochSpec:
    loss: object
    device_id: int


@dataclass
class OptimizerConfig:
    optimizer: torch.optim.Optimizer
    kwargs: dict


@dataclass
class SchedulerConfig:
    scheduler: type  # e.g. torch.optim.lr_scheduler.ExponentialLR
    kwargs: dict     # e.g. dict(gamma=0.98)


@dataclass
class TrainConfig:
    model_config: object
    train_data_config: object
    loss: torch.nn.Module
    optimizer: OptimizerConfig
    batch_size: int
    gradient_clipping: Union[None, float] = None
    scheduler_config: Union[None, SchedulerConfig] = None
    ensemble_id: int = 0
    _version: int = 0
    val_data_config: object = None
    post_model_create_hook: object = None
    model_pre_train_hook: object = None
    extra: object = None

    def serialize_human(self):
        import lib.model_factory as model_factory
        import lib.data_factory as data_factory

        val_data = None
        if self.val_data_config is not None:
            val_data = dict(
                config=lib.serialize_human.serialize_human(self.val_data_config),
                name=data_factory.get_factory()
                .get_class(self.val_data_config)
                .__name__,
            )
        return dict(
            model=dict(
                config=self.model_config.serialize_human(),
                name=model_factory.get_factory().get_class(self.model_config).__name__,
                post_model_create_hook=(
                    self.post_model_create_hook.__name__
                    if self.post_model_create_hook is not None
                    else None
                ),
                model_pre_train_hook=(
                    self.model_pre_train_hook.__name__
                    if self.model_pre_train_hook is not None
                    else None
                ),
            ),
            data=dict(
                config=self.train_data_config.serialize_human(),
                name=data_factory.get_factory()
                .get_class(self.train_data_config)
                .__name__,
            ),
            val_data=val_data,
            loss=self.loss.__class__.__name__,
            optimizer=dict(
                name=self.optimizer.optimizer.__name__,
                kwargs=self.optimizer.kwargs,
            ),
            batch_size=self.batch_size,
            ensemble_id=self.ensemble_id,
            extra=lib.serialize_human.serialize_human(self.extra),
            gradient_clipping=self.gradient_clipping,
            scheduler_config=(
                dict(name=self.scheduler_config.scheduler.__name__, kwargs=self.scheduler_config.kwargs)
                if self.scheduler_config is not None
                else None
            ),
            _version=self._version,
        )

    def ensemble_dict(self):
        ensemble_config = self.__dict__.copy()
        ensemble_config.pop("ensemble_id")
        return ensemble_config


@dataclass
class TrainEval:
    train_metrics: List[Callable[[], Metric]]
    validation_metrics: List[Callable[[], Metric]]
    data_visualizer: Callable[[object, TrainEpochState], None] = None
    log_gradient_norm: bool = False
    log_parameter_norm: bool = False
    log_sample_ids: bool = True  # per-step row in train_steps; off for tiny-batch high-step regimes
    diagnostics_interval: int = 10  # run diagnostics every N steps

    def serialize_human(self):
        return dict(
            train_metrics=[metric().name() for metric in self.train_metrics],
            val_metric=[metric().name() for metric in self.validation_metrics],
            log_gradient_norm=self.log_gradient_norm,
            log_parameter_norm=self.log_parameter_norm,
            log_sample_ids=self.log_sample_ids,
            diagnostics_interval=self.diagnostics_interval,
        )


def get_num_gpus():
    try:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        if world_size > 1:
            return world_size
    except ValueError:
        pass

    cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices is not None:
        return len([device for device in cuda_devices.split(",") if device.strip()])
    return 0


def is_distributed():
    try:
        if "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"]) > 1
    except ValueError:
        return False
    # A single orchestration process sees all allocated GPUs, then launches
    # the per-rank torchrun workers in lib.distributed_trainer.
    return get_num_gpus() > 1


def get_num_workers():
    if os.getenv("EP_DEBUG") is not None:
        return 0
    elif (cpus_on_node := os.getenv("SLURM_CPUS_ON_NODE")) is not None:
        try:
            print(f"Using {cpus_on_node} workers [SLURM_CPUS_ON_NODE]")
            return int(cpus_on_node)
        except ValueError:
            print(
                f"SLURM_CPUS_ON_NODE exists but can't do integer conversion ({cpus_on_node})"
            )
    return os.cpu_count()


@dataclass
class ComputeConfig:
    num_workers: int = field(default_factory=get_num_workers)
    distributed: bool = field(default_factory=is_distributed)
    num_gpus: int = field(default_factory=get_num_gpus)
    find_unused_parameters: bool = True

    def serialize_human(self):
        return self.__dict__


@dataclass
class GPUMonitorConfig:
    """Configuration for GPU monitoring during training."""

    enabled: bool = True
    sample_interval: float = 1.0  # How often to sample GPU stats (seconds)
    record_interval: float = 10.0  # How often to record metrics (seconds)

    def serialize_human(self):
        return self.__dict__


def get_slurm_id():
    return os.environ.get("SLURM_JOB_ID", "None")


def get_modules():
    import importlib.metadata

    modules = []
    for dist in sorted(
        importlib.metadata.distributions(), key=lambda x: x.metadata["Name"].lower()
    ):
        modules.append(f"{dist.metadata['Name']}=={dist.version}")
    return "\n".join(modules)


@dataclass
class TrainRun:
    project: str
    compute_config: ComputeConfig
    train_config: TrainConfig
    train_eval: TrainEval
    epochs: int
    save_nth_epoch: int
    validate_nth_epoch: int
    keep_epoch_checkpoints: bool = False
    keep_nth_epoch_checkpoints: int = 1
    visualize_terminal: bool = True
    visualize_interval_s: int = 10
    gpu_monitor: GPUMonitorConfig = field(default_factory=GPUMonitorConfig)
    notes: object = None
    post_validate_hook: object = None  # Callable(model, train_run, state, device) called at end of validate() on rank 0
    argv: str = " ".join(sys.argv)
    git_rev: str = get_rev()
    env: str = field(default_factory=lambda: os.environ.copy())
    slurm_jobid: str = get_slurm_id()
    modules: str = get_modules()
    run_id: int = field(default_factory=random_positive_i64)

    def custom_dict(self):
        serialize_dict = copy.deepcopy(self.__dict__)
        del serialize_dict["notes"]
        return serialize_dict

    def serialize_human(self):
        return dict(
            project=self.project,
            compute_config=self.compute_config.serialize_human(),
            train_config=self.train_config.serialize_human(),
            train_id=stable_hash_str(self.train_config),
            ensemble_id=stable_hash_str(self.train_config.ensemble_dict()),
            train_eval=self.train_eval.serialize_human(),
            epochs=self.epochs,
            save_nth_epoch=self.save_nth_epoch,
            visualize_terminal=self.visualize_terminal,
            visualize_interval_s=self.visualize_interval_s,
            gpu_monitor=self.gpu_monitor.serialize_human(),
            notes=self.notes,
            argv=self.argv,
            git_rev=self.git_rev,
            env=self.env,
            slurm_jobid=self.slurm_jobid,
            modules=self.modules,
            run_id=self.run_id,
        )


@dataclass
class EnsembleConfig:
    members: list[TrainRun]

    def serialize_human(self):
        return [member_config.serialize_human() for member_config in self.members]


@dataclass
class Ensemble:
    member_configs: list[TrainRun]
    members: list[torch.nn.Module]
    n_members: int

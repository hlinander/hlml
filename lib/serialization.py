import dataclasses
import importlib
import json
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lib.train_dataclasses import TrainEpochState
from lib.train_dataclasses import TrainRun, TrainConfig
import lib.data_factory as data_factory
from lib.timing_metric import Timing
from lib.paths import (
    get_checkpoint_path,
    get_or_create_checkpoint_path,
    get_model_epoch_checkpoint_path,
)
import lib.model_factory as model_factory
from lib.data_utils import get_sampler, make_dataloader
from lib.ddp import get_rank
from lib.stable_hash import json_dumps_dataclass_str
import shutil
import lib.render_duck as duck
from lib.log import log
from lib.compute_env import env

# Re-export for backward compatibility
from lib.serialize_human import serialize_human


@dataclass
class SerializeConfig:
    train_run: TrainRun
    train_epoch_state: TrainEpochState


@dataclass
class FileStructure:
    model_id: int = None
    model: object = None
    optimizer: object = None
    epoch: object = None
    batch: object = None
    train_metrics: object = None
    validation_metrics: object = None
    train_run: object = None
    timing_metric: object = None


def write_status_file(config: SerializeConfig):
    if get_rank() != 0:
        return

    checkpoint_path = get_or_create_checkpoint_path(config.train_run.train_config)
    with open(checkpoint_path / "status_tmp", "w") as status_file:
        status_file.write(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Epoch {config.train_epoch_state.epoch} / {config.train_run.epochs}, Batch {config.train_epoch_state.batch} / {config.train_run.epochs * len(config.train_epoch_state.train_dataloader)}"
        )
    shutil.move(checkpoint_path / "status_tmp", checkpoint_path / "status")


def serialize(config: SerializeConfig):
    if get_rank() != 0:
        print("I am not rank 0 so not serializing...")
        return
    train_config = config.train_run.train_config
    train_epoch_state = config.train_epoch_state
    train_run = config.train_run

    keep_checkpoint_condition = config.train_run.keep_epoch_checkpoints and (
        train_epoch_state.epoch % config.train_run.keep_nth_epoch_checkpoints == 0
    )
    not_serialize_condition = train_epoch_state.epoch % train_run.save_nth_epoch != 0

    # If neither full serialization nor model checkpoint requested, return early
    if (not keep_checkpoint_condition) and not_serialize_condition:
        # Save if this is the last epoch regardless
        if train_epoch_state.epoch != train_run.epochs:
            log(
                "serialize",
                f"Not serializing....{train_epoch_state.epoch},{train_run.epochs},{train_run.save_nth_epoch},{not_serialize_condition}",
            )
            return

    if config.train_run.compute_config.distributed:
        model = train_epoch_state.model.module.state_dict()
    else:
        model = train_epoch_state.model.state_dict()

    file_data = FileStructure(
        model_id=train_epoch_state.model_id,
        model=model,
        optimizer=train_epoch_state.optimizer.state_dict(),
        epoch=train_epoch_state.epoch,
        batch=train_epoch_state.batch,
        train_run=config.train_run.serialize_human(),
        timing_metric=train_epoch_state.timing_metric.serialize(),
    )

    checkpoint_path = get_or_create_checkpoint_path(train_config)

    full_checkpoint_path_str = None
    if config.train_run.keep_epoch_checkpoints and keep_checkpoint_condition:
        model_epoch_checkpoint = get_model_epoch_checkpoint_path(
            config.train_run.train_config, train_epoch_state.epoch
        )
        torch.save(model, f"{model_epoch_checkpoint}_tmp")
        full_checkpoint_path_str = model_epoch_checkpoint.absolute().as_posix()

        shutil.move(
            f"{model_epoch_checkpoint}_tmp",
            model_epoch_checkpoint,
        )
    for key, value in file_data.__dict__.items():
        torch.save(value, checkpoint_path / f"{key}_tmp")
        shutil.move(checkpoint_path / f"{key}_tmp", checkpoint_path / key)

    with open(checkpoint_path / "train_run.json_tmp", "w") as train_run_json_file:
        train_run_json_file.write(json_dumps_dataclass_str(config.train_run, indent=2))
    shutil.move(
        checkpoint_path / "train_run.json_tmp", checkpoint_path / "train_run.json"
    )
    write_status_file(config)
    duck.insert_checkpoint(
        train_epoch_state.model_id,
        train_epoch_state.batch,
        full_checkpoint_path_str,
    )
    log("serialize", f"Serialized at {checkpoint_path}")


def get_train_run_status(train_run: TrainRun):
    status_file = get_checkpoint_path(train_run.train_config) / "status"
    if status_file.is_file():
        try:
            return open(status_file).read()
        except:
            return "Couldn't read status..."
    else:
        return "No status file."


@dataclass
class DeserializeConfig:
    train_run: TrainRun
    device_id: torch.device


def is_serialized(train_run: TrainRun):
    train_config = train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config) / "model"
    return checkpoint_path.is_file()


def instantiate_model(train_config: TrainConfig):
    data_spec = (
        data_factory.get_factory()
        .get_class(train_config.train_data_config)
        .data_spec(train_config.train_data_config)
    )
    model = model_factory.get_factory().create(train_config.model_config, data_spec)
    return model


def create_model(config: DeserializeConfig, state_dict: torch.Tensor):
    train_config = config.train_run.train_config
    # data_spec = (
    #     data_factory.get_factory()
    #     .get_class(train_config.train_data_config)
    #     .data_spec(train_config.train_data_config)
    # )
    # model = model_factory.get_factory().create(train_config.model_config, data_spec)
    model = instantiate_model(config.train_run.train_config)
    model = model.to(device=torch.device(config.device_id))

    if config.train_run.compute_config.distributed:
        device = torch.device(config.device_id)
        ddp_kwargs = {"device_ids": [config.device_id]} if device.type == "cuda" else {}
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=getattr(
                config.train_run.compute_config, "find_unused_parameters", True
            ),
            **ddp_kwargs,
        )
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    if train_config.model_pre_train_hook is not None:
        model = train_config.model_pre_train_hook(model, config.train_run)

    return model


@dataclass
class DeserializedModel:
    model: torch.nn.Module
    model_id: int
    epoch: int


def deserialize_model(config: DeserializeConfig, latest_ok=False) -> DeserializedModel:
    train_config = config.train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config)
    print(f"Trying to load {checkpoint_path.absolute()}")
    if (
        not (checkpoint_path / "model").is_file()
        or not (checkpoint_path / "epoch").is_file()
        or not (checkpoint_path / "model_id").is_file()
    ):
        print("No files")
        return None
    else:
        print(f"{checkpoint_path}")

    try:
        model_epoch = torch.load(checkpoint_path / "epoch")
        print("model_epoch", model_epoch)
        model_id = torch.load(checkpoint_path / "model_id")

        if model_epoch != config.train_run.epochs and not latest_ok:
            model_epoch_checkpoint = get_model_epoch_checkpoint_path(
                config.train_run.train_config, config.train_run.epochs
            )
            if not (model_epoch_checkpoint).is_file():
                raise Exception(
                    f"The requested epoch ({config.train_run.epochs}) is not available in {model_epoch_checkpoint}."
                )
            model_epoch = config.train_run.epochs
            model_state_dict = torch.load(
                model_epoch_checkpoint, map_location=torch.device(config.device_id)
            )
            print(
                f"Loaded earlier epoch {config.train_run.epochs}, the latest epoch is {model_epoch}."
            )
        else:
            model_state_dict = torch.load(
                checkpoint_path / "model", map_location=torch.device(config.device_id)
            )

    except Exception as e:
        print(f"Failed to deserialize_model: {e}")
        return None

    return DeserializedModel(
        model=create_model(config, model_state_dict),
        epoch=model_epoch,
        model_id=model_id,
    )


def get_serialization_epoch(config: DeserializeConfig):
    return get_serialized_epoch(config.train_run.train_config)


def get_serialized_epoch(train_config: TrainConfig):
    checkpoint_path = get_checkpoint_path(train_config)
    if (
        not (checkpoint_path / "model").is_file()
        or not (checkpoint_path / "epoch").is_file()
    ):
        return None
    else:
        print(f"{checkpoint_path}")

    try:
        model_epoch = torch.load(checkpoint_path / "epoch")
    except Exception as e:
        print(f"Failed to get_serialization_epoch: {e}")
        return None

    return model_epoch


def deserialize(config: DeserializeConfig):
    train_config = config.train_run.train_config
    checkpoint_path = get_checkpoint_path(train_config)
    print("Trying to deserialize at", checkpoint_path)
    if not (checkpoint_path / "model").is_file():
        return None
    else:
        print(f"{checkpoint_path}")

    file_data = FileStructure()
    for key in list(file_data.__dict__.keys()):
        print(checkpoint_path / key)
        try:
            setattr(
                file_data,
                key,
                torch.load(
                    checkpoint_path / key, map_location=torch.device(config.device_id)
                ),
            )
        except RuntimeError as e:
            print(f"Deserialize failed: {e}")
            return None
        except EOFError as e:
            print(f"Deserialize failed: {e}")
            return None
        except KeyError as e:
            print(f"Deserialize failed: {e}")
            return None
        except Exception as e:
            print(f"Deserialize failed: {e}")
            return None
    data_dict = file_data.__dict__

    train_ds = data_factory.get_factory().create(train_config.train_data_config)
    train_dataloader = make_dataloader(
        train_ds,
        config.train_run,
        config.device_id,
        shuffle=True,
        seed=train_config.ensemble_id,
    )
    if train_config.val_data_config is not None:
        val_ds = data_factory.get_factory().create(train_config.val_data_config)
        val_dataloader = make_dataloader(
            val_ds,
            config.train_run,
            config.device_id,
            shuffle=False,
            seed=train_config.ensemble_id,
        )
    else:
        val_dataloader = None

    duck.LAST_MODEL_ID = file_data.model_id
    duck.LAST_RUN_CONFIG = config.train_run
    model = create_model(config, data_dict["model"])

    optimizer = train_config.optimizer.optimizer(
        model.parameters(), **train_config.optimizer.kwargs
    )
    optimizer.load_state_dict(data_dict["optimizer"])

    train_metrics = []
    for metric in config.train_run.train_eval.train_metrics:
        metric_instance = metric()
        train_metrics.append(metric_instance)

    validation_metrics = []
    for metric in config.train_run.train_eval.validation_metrics:
        metric_instance = metric()
        validation_metrics.append(metric_instance)

    timing_metric = Timing()
    timing_metric.deserialize(data_dict["timing_metric"])

    epoch = data_dict["epoch"]

    return TrainEpochState(
        model_id=file_data.model_id,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        batch=data_dict["batch"],
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        next_visualization=0.0,
        next_visualizer=0,
        timing_metric=timing_metric,
    )


def _get_config_class_registry():
    registry = {}
    registry.update(model_factory.get_factory().config_classes)
    registry.update(data_factory.get_factory().config_classes)
    return registry


def _resolve_callable_string(value, field_def=None):
    """Resolve a string back to a callable.

    Tries fully qualified import first (e.g. "torch.nn.modules.normalization.LayerNorm"),
    then falls back to known namespaces for legacy short names (e.g. "LayerNorm"),
    then tries matching the field default.
    """
    if not isinstance(value, str):
        return value

    # Try fully qualified import: "some.module.ClassName"
    if "." in value:
        module_path, _, attr_name = value.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            pass

    # Try known namespaces for legacy short names
    for namespace in [torch.nn, torch.optim]:
        obj = getattr(namespace, value, None)
        if obj is not None:
            return obj

    # Try matching the field default
    if field_def is not None:
        default = field_def.default
        if default is not dataclasses.MISSING and callable(default):
            if hasattr(default, "__name__") and default.__name__ == value:
                return default

    return value


def _deserialize_dataclass(json_dict, class_registry):
    """Reconstruct a dataclass instance from the __class__/__data__ format
    produced by serialize_dataclass."""
    class_name = json_dict["__class__"]
    cls = class_registry[class_name]
    fields = {f.name: f for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
    data = {}
    for k, v in json_dict["__data__"].items():
        if isinstance(v, dict) and "__class__" in v:
            data[k] = _deserialize_dataclass(v, class_registry)
        elif isinstance(v, list):
            data[k] = [
                _deserialize_dataclass(item, class_registry)
                if isinstance(item, dict) and "__class__" in item
                else item
                for item in v
            ]
        elif k in fields:
            data[k] = _resolve_callable_string(v, fields[k])
        else:
            data[k] = v
    return cls(**data)


def _read_checkpoint_path(checkpoint_hash_hex, checkpoint_root=None):
    root = (
        env().paths.checkpoints
        if checkpoint_root is None
        else Path(checkpoint_root)
    )
    return root / f"checkpoint_{checkpoint_hash_hex}"


def load_model_from_checkpoint(
    checkpoint_hash_hex,
    device_id,
    epoch=None,
    *,
    checkpoint_root=None,
):
    """Load a model from a checkpoint using the config saved in train_run.json.

    This allows loading checkpoints whose TrainConfig no longer exists in code,
    as long as the model and data config classes are still registered in the
    factories.

    Args:
        checkpoint_hash_hex: The hex hash string identifying the checkpoint
            (e.g. "a1b2c3d4e5f67890").
        device_id: The torch device to load the model onto.
        epoch: If provided, load the epoch-specific checkpoint
            (model_epoch_XXXX) instead of the latest model.
        checkpoint_root: Optional directory containing ``checkpoint_*``
            directories. The configured checkpoint root remains the default.

    Returns:
        A DeserializedModel, or None if the checkpoint doesn't exist.
    """
    checkpoint_path = _read_checkpoint_path(
        checkpoint_hash_hex, checkpoint_root
    )
    json_path = checkpoint_path / "train_run.json"
    if not json_path.is_file():
        print(f"No train_run.json found at {checkpoint_path}")
        return None

    saved = json.loads(json_path.read_text())
    train_config_data = saved["__data__"]["train_config"]["__data__"]
    registry = _get_config_class_registry()

    model_config = _deserialize_dataclass(
        train_config_data["model_config"], registry
    )
    data_config = _deserialize_dataclass(
        train_config_data["train_data_config"], registry
    )

    data_spec = (
        data_factory.get_factory()
        .get_class(data_config)
        .data_spec(data_config)
    )
    model = model_factory.get_factory().create(model_config, data_spec)
    model = model.to(device=torch.device(device_id))

    model_id = torch.load(
        checkpoint_path / "model_id", map_location=torch.device(device_id)
    )

    if epoch is not None:
        model_path = checkpoint_path / f"model_epoch_{epoch:04d}"
    else:
        model_path = checkpoint_path / "model"

    if not model_path.is_file():
        print(f"No model file found at {model_path}")
        return None

    loaded_epoch = epoch if epoch is not None else torch.load(
        checkpoint_path / "epoch", map_location=torch.device(device_id)
    )
    state_dict = torch.load(model_path, map_location=torch.device(device_id))
    model.load_state_dict(state_dict)

    return DeserializedModel(model=model, epoch=loaded_epoch, model_id=model_id)


def load_checkpoint_train_run_json(checkpoint_hash_hex, *, checkpoint_root=None):
    """Load the raw train_run.json dict from a checkpoint."""
    checkpoint_path = _read_checkpoint_path(
        checkpoint_hash_hex, checkpoint_root
    )
    return json.loads((checkpoint_path / "train_run.json").read_text())


def load_checkpoint_data_config(checkpoint_hash_hex, *, checkpoint_root=None):
    """Load the train data config from a checkpoint's saved train_run.json."""
    checkpoint_path = _read_checkpoint_path(
        checkpoint_hash_hex, checkpoint_root
    )
    json_path = checkpoint_path / "train_run.json"
    saved = json.loads(json_path.read_text())
    train_config_data = saved["__data__"]["train_config"]["__data__"]
    registry = _get_config_class_registry()
    return _deserialize_dataclass(
        train_config_data["train_data_config"], registry
    )


def list_checkpoint_epochs(checkpoint_hash_hex, *, checkpoint_root=None):
    """List available epoch checkpoints for a given checkpoint hash.

    Returns a sorted list of epoch numbers that have saved model files.
    """
    checkpoint_path = _read_checkpoint_path(
        checkpoint_hash_hex, checkpoint_root
    )
    epochs = []
    for p in checkpoint_path.glob("model_epoch_*"):
        try:
            epoch = int(p.name.split("_")[-1])
            epochs.append(epoch)
        except ValueError:
            continue
    return sorted(epochs)

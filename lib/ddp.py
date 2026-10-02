from datetime import timedelta
import os
import torch


def get_rank() -> int:
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        return 0


def get_world_size() -> int:
    try:
        return int(os.environ["WORLD_SIZE"])
    except KeyError:
        return 1
    except ValueError:
        return 1


_DEVICE = None


def ddp_setup(backend=None) -> str:
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    global _DEVICE
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # breakpoint()
    if _DEVICE is not None:
        return _DEVICE

    if "EP_TORCHRUN" in os.environ:
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend, init_method="env://", timeout=timedelta(seconds=300)
            )
        device_id = int(os.environ["LOCAL_RANK"])
        if backend == "nccl":
            torch.cuda.set_device(device_id)
            _DEVICE = device_id
            return device_id
        if backend == "gloo":
            _DEVICE = "cpu"
            return "cpu"

        raise Exception("Unsupported backend")

    if "TORCH_DEVICE" in os.environ:
        device = os.environ["TORCH_DEVICE"]
        print(f"Using device {device}")
        _DEVICE = device
        return device

    print("No ep torch!")
    _DEVICE = "cpu"
    return "cpu"

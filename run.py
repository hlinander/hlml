#!/usr/bin/env -S uv run
"""
Unified experiment runner.

Usage:
    uv run python run.py experiments/mnist/dense.py              # auto-detect
    uv run python run.py --mode torchrun experiments/mnist/dense.py
    uv run python run.py --mode mps experiments/mnist/dense.py
    uv run python run.py --mode cpu experiments/mnist/dense.py
    uv run python run.py --mode debug experiments/mnist/dense.py
"""
import argparse
import os
import sys
import subprocess
import resource
from dataclasses import dataclass
from lib.compute_env import env as compute_env


@dataclass
class RunConfig:
    """Configuration for a run mode"""

    device: str | None
    runner: list[str]  # Command prefix
    env_vars: dict[str, str]


def get_mode_config(mode: str, nproc: int) -> RunConfig:
    """Get configuration for each mode"""
    configs = {
        "torchrun": RunConfig(
            # Let each torchrun worker select its device from LOCAL_RANK in
            # ddp_setup(). Setting TORCH_DEVICE here bypasses process-group
            # initialization and sends every worker to the same device.
            device=None,
            runner=[
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=1",
                f"--nproc_per_node={nproc}",
                "--rdzv_backend=c10d",
                "--rdzv_endpoint=localhost:0",
            ],
            env_vars={"EP_TORCHRUN": "1"},
        ),
        "cuda": RunConfig(
            device="cuda",
            runner=[sys.executable],
            env_vars={},
        ),
        "mps": RunConfig(
            device="mps",
            runner=[sys.executable],
            env_vars={},
        ),
        "cpu": RunConfig(
            device="cpu",
            runner=[sys.executable],
            env_vars={},
        ),
        "debug": RunConfig(
            device="cuda",
            runner=[sys.executable, "-m", "ipdb"],
            env_vars={},
        ),
    }
    return configs[mode]


def infer_torchrun_processes() -> int:
    """Infer the local worker count from the GPUs visible to this process."""
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        devices = [device for device in visible_devices.split(",") if device.strip()]
        if devices:
            return len(devices)

    try:
        import torch

        return max(torch.cuda.device_count(), 1)
    except ImportError:
        return 1


def detect_default_mode() -> str:
    """Auto-detect the best mode based on available hardware"""
    try:
        import torch

        if torch.cuda.is_available():
            return "torchrun"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments",
        usage="%(prog)s [--mode MODE] script.py [script_args...]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["auto", "torchrun", "cuda", "mps", "cpu", "debug"],
        default="auto",
        help="Run mode (default: auto-detect)",
    )
    parser.add_argument(
        "--nproc",
        "-n",
        type=int,
        default=None,
        help="Number of processes for torchrun (default: number of visible GPUs)",
    )
    parser.add_argument(
        "script",
        help="Python script to run",
    )

    args, remaining = parser.parse_known_args()

    # Resolve auto mode
    mode = args.mode if args.mode != "auto" else detect_default_mode()
    nproc = args.nproc
    if nproc is None:
        nproc = infer_torchrun_processes() if mode == "torchrun" else 1
    config = get_mode_config(mode, nproc)

    # Set file descriptor limit
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(64000, hard), hard))
    except (ValueError, resource.error):
        pass

    # Build environment
    env = os.environ.copy()
    if config.device is None:
        env.pop("TORCH_DEVICE", None)
    else:
        env["TORCH_DEVICE"] = config.device
    env["PYTHONBREAKPOINT"] = "ipdb.set_trace"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(config.env_vars)
    if compute_env().envs is not None:
        env.update(compute_env().envs)

    # Build command
    cmd = config.runner + [args.script] + remaining

    # Print info
    print(f"[run] Mode: {mode}, Device: {config.device}")
    print(f"[run] {' '.join(cmd)}")
    print()

    # Run
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

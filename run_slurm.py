"""
Submit a single experiment to SLURM via sbatch.

Usage:
    python run_slurm.py experiments/some_experiment.py           # submit
    python run_slurm.py --dry-run experiments/some_experiment.py # print script only
    python run_slurm.py experiments/some_experiment.py --extra-arg value
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from lib.slurm import (
    generate_batch_script,
    submit_batch_script,
    load_slurm_config_from_env,
)


def main():
    parser = argparse.ArgumentParser(
        description="Submit a single experiment to SLURM",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated batch script without submitting",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="SLURM job name (default: script stem)",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Optional dependency group to include (e.g. --extra graphcast)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override the number of GPUs requested from SLURM",
    )
    parser.add_argument(
        "--time",
        default=None,
        help="Override the SLURM time limit (for example, 00:10:00)",
    )
    parser.add_argument(
        "--constraint",
        default=None,
        help="Override the SLURM node constraint (for example, fat)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help="Override the number of CPUs allocated to the task",
    )
    parser.add_argument(
        "--direct-torchrun",
        action="store_true",
        help=(
            "Launch the script itself on every GPU instead of using the "
            "single-process training orchestrator"
        ),
    )
    parser.add_argument("script", help="Python script to run")

    args, remaining = parser.parse_known_args()

    slurm = load_slurm_config_from_env()
    slurm_overrides = {}
    if args.gpus is not None:
        slurm_overrides["gpus"] = args.gpus
    if args.time is not None:
        slurm_overrides["time"] = args.time
    if args.constraint is not None:
        slurm_overrides["constraint"] = args.constraint
    if args.cpus_per_task is not None:
        slurm_overrides["cpus_per_task"] = args.cpus_per_task
    if slurm_overrides:
        slurm = replace(slurm, **slurm_overrides)
    job_name = args.job_name or Path(args.script).stem

    extras = [f"--extra {e}" for e in args.extra]
    run_mode = "torchrun" if args.direct_torchrun else "cuda"
    run_parts = (
        ["uv run --no-sync"]
        + extras
        + ["python run.py", "--mode", run_mode, args.script]
        + remaining
    )
    run_command = " ".join(run_parts)

    script = generate_batch_script(
        job_name=job_name,
        slurm=slurm,
        run_command=run_command,
    )

    submit_batch_script(script, slurm, args.dry_run)


if __name__ == "__main__":
    main()

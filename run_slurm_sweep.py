"""
SLURM sweep runner.

Submit mode (default): generate a SLURM batch script and submit via sbatch.
Worker mode (--worker): execute a single config by array index.

Usage:
    python run_sweep.py experiments/some_sweep.py              # submit
    python run_sweep.py --dry-run experiments/some_sweep.py    # print script only
    python run_sweep.py --run-local experiments/some_sweep.py  # run all locally
    python run_sweep.py --worker experiments/some_sweep.py 3   # run config 3
"""

import argparse
import importlib.util
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

from lib.slurm import (
    SlurmConfig,
    generate_batch_script,
    submit_batch_script,
    load_slurm_config_from_env,
)


def load_sweep_module(sweep_path: str):
    """Import a sweep file as a module."""
    spec = importlib.util.spec_from_file_location("sweep", sweep_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sweep module from {sweep_path}")
    module = importlib.util.module_from_spec(spec)
    # Some decorators (notably ``dataclasses.dataclass`` on Python 3.11)
    # resolve annotations through ``sys.modules[cls.__module__]`` while the
    # module body is executing.  Mirror normal import semantics by publishing
    # the module before calling its loader.
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    return module


def load_slurm_config(sweep_module) -> SlurmConfig:
    """Load SlurmConfig from sweep module, falling back to env.py, then defaults."""
    if hasattr(sweep_module, "get_slurm_config"):
        return sweep_module.get_slurm_config()
    return load_slurm_config_from_env()


def cmd_submit(
    sweep_path: str,
    dry_run: bool,
    max_concurrent: int | None,
    extras: list[str] = [],
    dependency: str | None = None,
    array_start: int | None = None,
    array_end: int | None = None,
    deferred_num_configs: int | None = None,
):
    """Submit mode: generate batch script and submit via sbatch."""
    module = load_sweep_module(sweep_path)
    if deferred_num_configs is None:
        configs = module.create_configs()
        num_configs = len(configs)
    else:
        if dependency is None:
            raise ValueError(
                "--deferred-num-configs requires --dependency so workers cannot "
                "start before their manifests exist"
            )
        if deferred_num_configs < 1:
            raise ValueError("--deferred-num-configs must be positive")
        num_configs = deferred_num_configs
    print(f"[sweep] Found {num_configs} configs in {sweep_path}")
    if deferred_num_configs is not None:
        print(
            "[sweep] Config count is deferred; each worker will validate its "
            "index after the dependency succeeds"
        )

    slurm = load_slurm_config(module)
    if dependency is not None:
        slurm = replace(
            slurm,
            extra_sbatch=[
                *slurm.extra_sbatch,
                f"--dependency=afterok:{dependency}",
            ],
        )

    first_index = 0 if array_start is None else array_start
    last_index = num_configs - 1 if array_end is None else array_end
    if first_index < 0 or last_index >= num_configs or first_index > last_index:
        raise ValueError(
            f"invalid array slice {first_index}-{last_index} for "
            f"{num_configs} configs"
        )
    # Slurm constrains array *task ids* to ``MaxArraySize - 1`` even when the
    # number of tasks is smaller.  Use a zero-based physical array for a
    # logical slice and add the frozen config offset in the worker command.
    array_spec = f"0-{last_index - first_index}"
    if max_concurrent is not None:
        array_spec += f"%{max_concurrent}"

    module_extras = list(getattr(module, "UV_EXTRAS", ()))
    selected_extras = list(dict.fromkeys([*module_extras, *extras]))
    extra_flags = " ".join(f"--extra {e}" for e in selected_extras)
    # Array workers share one immutable project environment.  Never let a
    # compute task resolve or mutate it: that serializes unrelated workers on
    # the uv environment lock and is unsafe when another worktree shares the
    # same .venv.  Dependency changes belong in an explicit setup step.
    uv_run = f"uv run --no-sync {extra_flags}".strip()

    script = generate_batch_script(
        job_name=Path(sweep_path).stem,
        slurm=slurm,
        run_command=(
            f"{uv_run} python run.py --mode cuda run_slurm_sweep.py "
            f"--worker {sweep_path} "
            f"$((SLURM_ARRAY_TASK_ID + {first_index}))"
        ),
        array_spec=array_spec,
    )

    submit_batch_script(script, slurm, dry_run)


def cmd_worker(sweep_path: str, task_index: int):
    """Worker mode: run a single config by index."""
    module = load_sweep_module(sweep_path)
    configs = module.create_configs()
    if task_index < 0 or task_index >= len(configs):
        print(
            f"[sweep] Error: task index {task_index} out of range "
            f"(0-{len(configs) - 1})",
            file=sys.stderr,
        )
        sys.exit(1)
    config = configs[task_index]()
    module.run(config)


def cmd_run_local(sweep_path: str):
    """Run all configs sequentially without SLURM."""
    # os.environ["EP_DEBUG"] = "1"
    module = load_sweep_module(sweep_path)
    configs = module.create_configs()
    print(f"[sweep] Running {len(configs)} configs locally")
    for i, factory in enumerate(configs):
        print(f"[sweep] Config {i}/{len(configs) - 1}")
        config = factory()
        module.run(config)
    print("[sweep] Done")


def main():
    parser = argparse.ArgumentParser(
        description="SLURM sweep runner",
        usage=textwrap.dedent(
            """\
            %(prog)s [options] sweep_file.py
            %(prog)s --worker sweep_file.py INDEX"""
        ),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Worker mode: run a single config by index",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated batch script without submitting",
    )
    parser.add_argument(
        "--run-local",
        action="store_true",
        help="Run all configs sequentially without SLURM",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None, help="Limit SLURM array concurrency"
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Optional dependency group to include (e.g. --extra graphcast)",
    )
    parser.add_argument(
        "--dependency",
        help="Release the array only after this Slurm job succeeds",
    )
    parser.add_argument(
        "--array-start",
        type=int,
        help="First frozen config index to submit (default: 0)",
    )
    parser.add_argument(
        "--array-end",
        type=int,
        help="Last frozen config index to submit (default: final config)",
    )
    parser.add_argument(
        "--deferred-num-configs",
        type=int,
        help=(
            "Known future config count; skip create_configs during submission. "
            "Requires --dependency and validates the real configs in each worker."
        ),
    )
    parser.add_argument("sweep_file", help="Path to sweep file")
    parser.add_argument(
        "task_index", nargs="?", type=int, help="Task index (worker mode only)"
    )

    args = parser.parse_args()

    if args.worker:
        if args.task_index is None:
            parser.error("worker mode requires a task index")
        cmd_worker(args.sweep_file, args.task_index)
    elif args.run_local:
        cmd_run_local(args.sweep_file)
    else:
        cmd_submit(
            args.sweep_file,
            args.dry_run,
            args.max_concurrent,
            args.extra,
            args.dependency,
            args.array_start,
            args.array_end,
            args.deferred_num_configs,
        )


if __name__ == "__main__":
    main()

[![CI](https://github.com/hlinander/hlml/actions/workflows/main.yml/badge.svg)](https://github.com/hlinander/hlml/actions/workflows/main.yml)

# HLML

Train and evaluate PyTorch models with reproducibility in mind.

- Computational environment reproducible through Nix flake.
- Python based configuration in terms of dataclasses.
- Convenient metric functionality with focus on saving as much as possible for future inspection.
- Simple TUI for easy progress inspection.
- DuckDB-based storage with S3 ingestion pipeline for scalable run tracking and visualization.


## Quick Start with uv

Install [uv](https://github.com/astral-sh/uv) and run experiments:

```bash
uv run run.py experiments/mnist/dense.py            # auto-detect device
uv run run.py --mode cuda experiments/mnist/dense.py
uv run run.py --mode cpu experiments/mnist/dense.py
```

On first run, `env.py` is auto-created with defaults. All data is stored under `.local/` (add to `.gitignore`).

### Analytics credentials

To upload analytics to S3 staging you need your own S3 credentials. 

```
~/.config/equivariant-posteriors/analytics.yaml   # your credentials (out of tree)
analytics.example.yaml                            # committed template
```

Setup (once):

```bash
mkdir -p ~/.config/equivariant-posteriors
cp analytics.example.yaml ~/.config/equivariant-posteriors/analytics.yaml
$EDITOR ~/.config/equivariant-posteriors/analytics.yaml   # paste your key/secret + endpoint/bucket
```

Then enable S3 staging in your `env.py` (uncomment Option 2 in `env.example.py`),
which loads the file via `lib/analytics_credentials.py`:

```python
from lib.analytics_credentials import load_staging_s3
staging = load_staging_s3()
```

Override the file location with `ANALYTICS_CREDENTIALS_FILE` if needed (e.g. CI).

### Feed analytics target

Feed can replace legacy Parquet staging while keeping the local DuckDB and
checkpoint analytics used for resume and terminal visualization:

```bash
uv sync
uv run feed login https://eqp.hampe.nu/ingest
```

Login automatically selects the feed when only one is available. Otherwise,
select the default once with `uv run feed use project/feed`.

Enable Feed in `env.py`:

```python
from lib.analytics_config import AnalyticsConfig, CentralDuckDB, FeedTarget

def get_analytics_config():
    return AnalyticsConfig(
        staging=FeedTarget(
            server_url="https://eqp.hampe.nu/ingest",
        ),
        central=CentralDuckDB(),  # Not used by a Feed training client.
        export_interval_seconds=2,
    )
```

Pass `feed="project/feed"` to `FeedTarget` to override the saved default.
The existing `project=` argument remains a compatibility alias. Set `spool_dir`
to a durable private directory with working file locks, atomic replacement and
fsync. Use a separate spool when upgrading clients: newer Feed spool formats
may not be readable by an older client. Node-local temporary spools do not
survive removal of that node's temporary storage.

The recurring exporter sends timestamp-safe chunks, waits for acknowledged
Feed delivery, and only then advances its independent local cursors. Run
configuration appears in `runs`; dynamic parameters in `model_parameters`;
scalar curves in `metric`; richer data in `epoch_metrics`,
`evaluation_samples`, `train_steps`, and `checkpoints`.
Heterogeneous values retain their native scalar types in columns such as
`value_int`, `value_float`, and `value_text`, selected by `value_type`.

The pinned client receives run metadata as an explicit `runs` event, including
the nested configuration and identifiers needed to join metrics. Local
checkpoint persistence is not a remote acknowledgement. Failed or incomplete
delivery leaves the Feed cursor unchanged and retains the local source rows.
Recovery is at least once, not exactly once: replay without a saved Feed cursor
can duplicate records. Inspect retained spools with `feed status` and reconcile
before `feed sync` or a backfill; this integration does not automatically migrate
old spools or delete source metrics.

### SLURM

Submit a single experiment to SLURM:

```bash
python run_slurm.py experiments/weather/train.py
python run_slurm.py --dry-run experiments/weather/train.py  # print script only
```

Submit a sweep (array job) to SLURM:

```bash
python run_slurm_sweep.py experiments/weather/sweep.py
python run_slurm_sweep.py --dry-run experiments/weather/sweep.py
python run_slurm_sweep.py --max-concurrent 4 experiments/weather/sweep.py
python run_slurm_sweep.py --run-local experiments/weather/sweep.py  # test locally
```

A sweep file defines `create_configs()` returning a list of callables and `run(config)`:

```python
def create_configs():
    return [lambda lr=lr: {"lr": lr} for lr in [1e-2, 1e-3, 1e-4]]

def run(config):
    print(f"Training with {config}")
```

SLURM parameters (time, GPUs, partition, etc.) are configured via `get_slurm_config()` in `env.py`. Sweep files can override this by defining their own `get_slurm_config()`.

Install the required dependencies with `uv sync` before submitting jobs. Workers
use `uv run --no-sync` so concurrent jobs do not modify their shared environment.
Single jobs accept `--gpus`, `--time`, `--constraint`, `--cpus-per-task` and
`--direct-torchrun`. Sweeps accept `--dependency JOBID`, `--array-start` and
`--array-end`; `--deferred-num-configs` requires a success dependency.

### Local Ingestion

After training, ingest metrics into the central database for querying:

```bash
uv run python ingestion/ingest.py
```

Query results with DuckDB:

```python
import duckdb
conn = duckdb.connect(".local/analytics.db")
conn.execute("SELECT * FROM train_step_metric_float WHERE name = 'loss'").fetchdf()
```

## Nix
This project uses [Nix](https://nixos.org/) for reproducible compute environmens.

### Binary cache from [Cachix](https://www.cachix.org/)
We provide cached builds of dependencies for a CUDA enabled system through Cachix. See instructions at
[https://app.cachix.org/cache/equivariant-posteriors#pull](https://app.cachix.org/cache/equivariant-posteriors#pull).

It is probably also a good idea to enable the CUDA maintainers cache:
https://app.cachix.org/cache/cuda-maintainers#pull

### Install
Install nix to your home-folder or system wide.
```
  https://nixos.org/download.html#download-nix
```
Enable [flakes](https://zero-to-nix.com/concepts/flakes) by one of

 1. Nix in other distribution: Create (or add to) `~/.config/nix/nix.conf` (or `/etc/nix/nix.conf`)
    ```
      experimental-features = nix-command flakes
    ```
 2. NixOS: Add to `/etc/nixos/configuration.nix`
    ```
      nix = {
        package = pkgs.nixFlakes;
        extraOptions = ''
          experimental-features = nix-command flakes
        '';
      };
    ```


### Development 
Start a development shell
```
  nix develop
```

### Test project
To build and run tests
```
  nix build
```

### Build singularity image
Build a singularity image with CUDA support containing the project
```
  nix build .#sing
```

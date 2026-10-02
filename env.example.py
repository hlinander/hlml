"""
Example environment configuration.

Copy this to env.py and customize for your deployment.
On first run, env.py is auto-created with local filesystem defaults.
"""
from pathlib import Path
from lib.analytics_config import (
    AnalyticsConfig,
    S3Config,
    PostgresConfig,
    StagingS3,
    StagingFilesystem,
    FeedTarget,
    CentralDuckDB,
    CentralDuckLake,
)
from lib.compute_env_config import ComputeEnvironment, Paths
from lib.slurm import SlurmConfig


# Base directory for all local data
LOCAL_DIR = Path(".local")


def get_env() -> ComputeEnvironment:
    """Configure paths for checkpoints, artifacts, etc."""
    return ComputeEnvironment(
        paths=Paths(
            checkpoints=LOCAL_DIR / "checkpoints",
            locks=LOCAL_DIR / "locks",
            distributed_requests=LOCAL_DIR / "distributed_requests",
            artifacts=LOCAL_DIR / "artifacts",
            datasets=LOCAL_DIR / "datasets",
        ),
    )


def get_slurm_config() -> SlurmConfig:
    """Configure SLURM job parameters for sweep submission."""
    return SlurmConfig(
        time="24:00:00",
        gpus=1,
    )


def get_analytics_config() -> AnalyticsConfig:
    """
    Configure the analytics pipeline.

    Flow: Training Client → Staging → Central Database
    """

    # -------------------------------------------------------------------------
    # Option 1: Local Filesystem (default)
    # -------------------------------------------------------------------------
    return AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=LOCAL_DIR / "staging",
            archive_dir=LOCAL_DIR / "archive",
        ),
        central=CentralDuckDB(
            db_path=LOCAL_DIR / "analytics.db",
        ),
    )

    # -------------------------------------------------------------------------
    # Option 2: S3 Staging + Local DB (upload analytics / test S3)
    # -------------------------------------------------------------------------
    # Credentials are read from ~/.config/equivariant-posteriors/analytics.yaml
    # (outside the repo, so they can never be committed). Copy the template:
    #
    #     mkdir -p ~/.config/equivariant-posteriors
    #     cp analytics.example.yaml ~/.config/equivariant-posteriors/analytics.yaml
    #     $EDITOR ~/.config/equivariant-posteriors/analytics.yaml
    #
    # See the "Analytics credentials" section of the README for details.
    #
    # from lib.analytics_credentials import load_staging_s3
    # return AnalyticsConfig(
    #     staging=load_staging_s3(),
    #     central=CentralDuckDB(
    #         db_path=LOCAL_DIR / "analytics.db",
    #     ),
    # )

    # -------------------------------------------------------------------------
    # Option 3: S3 Staging + DuckLake (central ingest — operator only)
    # -------------------------------------------------------------------------
    # The same home-folder credentials feed both the staging upload and the
    # DuckLake S3 data backend. The Postgres metadata password is an operator
    # secret managed outside this repo.
    #
    # from lib.analytics_credentials import load_staging_s3, s3_config
    # return AnalyticsConfig(
    #     staging=load_staging_s3(),
    #     central=CentralDuckLake(
    #         postgres=PostgresConfig(
    #             host="postgres.example.com",
    #             password="your-password",
    #         ),
    #         s3=s3_config(),
    #         data_path="s3://ducklake-data",
    #     ),
    #     export_interval_seconds=300,
    #     ingest_interval_seconds=300,
    # )

    # -------------------------------------------------------------------------
    # Option 4: Feed (low-latency project-scoped ingest)
    # -------------------------------------------------------------------------
    # Authenticate once with
    # `feed login https://eqp.hampe.nu/ingest`; credentials in a shared home
    # directory are available to jobs launched from the same environment. The
    # sole feed is automatic; use `feed use project/feed` when several exist.
    # return AnalyticsConfig(
    #     staging=FeedTarget(
    #         server_url="https://eqp.hampe.nu/ingest",
    #         # feed="project/feed",  # Override the saved default if needed.
    #         # spool_dir="/private/durable/feed-spool",  # Separate from older clients.
    #     ),
    #     central=CentralDuckDB(),  # Unused by training clients in Feed mode.
    #     export_interval_seconds=2,
    # )

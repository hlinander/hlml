"""
Unified export module using AnalyticsConfig.

Dispatches to either S3 or filesystem staging based on configuration.
"""
import threading
from typing import Optional
from lib.train_dataclasses import TrainRun
from lib.analytics_config import analytics_config, AnalyticsConfig
from lib.log import log
import lib.render_duck as duck


EXPORT_LOCK = threading.Lock()


def export_all(train_run: TrainRun, config: Optional[AnalyticsConfig] = None) -> list:
    """
    Export all metric tables using the configured staging backend.

    Args:
        train_run: The training run context
        config: Analytics configuration (defaults to analytics_config())

    Returns:
        List of paths (S3 or filesystem) that were created
    """
    with EXPORT_LOCK:
        return _export_all_unlocked(train_run, config)


def _export_all_unlocked(
    train_run: TrainRun, config: Optional[AnalyticsConfig] = None
) -> list:
    if config is None:
        config = analytics_config()

    if duck.CONN is None:
        log("export", "Error: DuckDB connection not initialized")
        return []

    # Feed serializes access to the existing connection. A duplicate DuckDB
    # connection can stall against EQP's attached in-memory `local` database.
    # The legacy Parquet exporters retain their established cursor behavior.
    cursor = duck.CONN if config.is_feed() else duck.CONN.cursor()

    # Dispatch based on staging type
    if config.is_s3_staging():
        from lib.export_parquet import flush_all_to_s3
        from lib.export_parquet import ensure_s3_credentials

        ensure_s3_credentials(cursor)

        paths = flush_all_to_s3(
            train_run=train_run,
            s3_bucket=config.staging.bucket,
            cursor=cursor,
            s3_prefix=config.staging.prefix,
        )

    elif config.is_filesystem_staging():
        from lib.staging_filesystem import flush_all_to_filesystem

        paths = flush_all_to_filesystem(
            train_run=train_run,
            staging_dir=config.staging.staging_dir,
            cursor=cursor,
        )

    elif config.is_feed():
        from lib.export_feed import finish_feed_export
        from lib.export_feed import _flush_checkpoint_analytics

        # Preserve the source before initialization, admission or HTTP can fail.
        with duck.CONN_LOCK:
            _flush_checkpoint_analytics(train_run, cursor)

        reference = finish_feed_export(
            train_run=train_run,
            target=config.staging,
            cursor=cursor,
        )
        paths = [reference] if reference is not None else []

    else:
        raise ValueError(f"Unknown staging type: {config.staging.type}")

    # Also flush to checkpoint directory for local persistence
    from lib.staging_filesystem import flush_all_to_checkpoint
    from lib.paths import get_or_create_checkpoint_path

    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    if config.is_feed():
        with duck.CONN_LOCK:
            ckpt_paths = flush_all_to_checkpoint(train_run, checkpoint_path, cursor)
    else:
        ckpt_paths = flush_all_to_checkpoint(train_run, checkpoint_path, cursor)
    if ckpt_paths:
        log(
            "export",
            f"Also saved {len(ckpt_paths)} parquet files to {checkpoint_path / 'analytics'}",
        )

    return paths


def start_periodic_export(
    train_run: TrainRun,
    config: Optional[AnalyticsConfig] = None,
    interval_seconds: Optional[int] = None,
):
    """
    Start a background thread that periodically exports data.

    Args:
        train_run: The training run context
        config: Analytics configuration (defaults to analytics_config())
        interval_seconds: Override export interval from config

    Returns:
        The background thread handle
    """
    if config is None:
        config = analytics_config()

    if interval_seconds is None:
        interval_seconds = config.export_interval_seconds

    # Dispatch based on staging type
    if config.is_s3_staging():
        from lib.export_parquet import export_periodic

        return export_periodic(
            train_run=train_run,
            interval_seconds=interval_seconds,
            s3_bucket=config.staging.bucket,
        )

    elif config.is_filesystem_staging():
        from lib.staging_filesystem import export_periodic_filesystem

        return export_periodic_filesystem(
            train_run=train_run,
            staging_dir=config.staging.staging_dir,
            interval_seconds=interval_seconds,
        )

    elif config.is_feed():
        from lib.export_feed import export_periodic_feed

        return export_periodic_feed(
            train_run=train_run,
            target=config.staging,
            interval_seconds=interval_seconds,
        )

    else:
        raise ValueError(f"Unknown staging type: {config.staging.type}")

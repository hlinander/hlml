"""
Export client-side DuckDB data to S3 as Parquet files for ingestion
"""
from pathlib import Path
from typing import Optional
import time
import duckdb
from lib.train_dataclasses import TrainRun
from lib.compute_env import env
from lib.log import log
import lib.render_duck as duck
from lib.render_duck import (
    MODEL_PARAMETER,
    TRAIN_STEP_METRIC,
    TRAIN_EPOCH_METRIC,
    CHECKPOINT_SAMPLE_METRIC,
    MODELS_TABLE_NAME,
    RUNS_TABLE_NAME,
    TRAIN_STEPS_TABLE_NAME,
    CHECKPOINTS_TABLE_NAME,
    ARTIFACTS_TABLE_NAME,
    ARTIFACT_CHUNKS_TABLE_NAME,
)

# Tables to sync
SYNC_TABLES = [
    MODEL_PARAMETER,
    TRAIN_STEP_METRIC,
    TRAIN_EPOCH_METRIC,
    CHECKPOINT_SAMPLE_METRIC,
    MODELS_TABLE_NAME,
    RUNS_TABLE_NAME,
    TRAIN_STEPS_TABLE_NAME,
    CHECKPOINTS_TABLE_NAME,
    ARTIFACTS_TABLE_NAME,
    ARTIFACT_CHUNKS_TABLE_NAME,
]


def flush_table_to_s3(
    table_name: str,
    s3_bucket: str,
    s3_prefix: str,
    cursor,
    run_id: Optional[int] = None,
) -> Optional[str]:
    """
    Flush a single table's new data to S3 as Parquet

    Args:
        table_name: Name of the table to export
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix for staging files
        cursor: Thread-safe DuckDB cursor
        run_id: Optional training run ID (filters by run_id if provided)

    Returns:
        S3 path if data was exported, None if no new data
    """
    # Ensure we're using the local database
    cursor.execute("USE local")

    # Get last synced timestamp
    result = cursor.execute(
        """
        SELECT COALESCE(MAX(last_synced_timestamp), 0) as last_synced
        FROM sync_state
        WHERE table_name = ?
        """,
        (table_name,),
    ).fetchall()

    last_synced = float(result[0][0]) if result and result[0][0] is not None else 0.0
    current_time = time.time()

    # Build WHERE clause based on whether we have run_id
    if run_id is not None:
        where_clause = "WHERE run_id = ? AND EPOCH(timestamp) > ?"
        where_params = (run_id, last_synced)
    else:
        where_clause = "WHERE EPOCH(timestamp) > ?"
        where_params = (last_synced,)

    # Check if there's new data to sync
    count_result = cursor.execute(
        f"""
        SELECT COUNT(*) FROM {table_name}
        {where_clause}
        """,
        where_params,
    ).fetchall()

    if not count_result or count_result[0][0] == 0:
        log("export", f"No new data to sync for {table_name}")
        return None

    row_count = count_result[0][0]

    # Use microseconds for timestamp to avoid collisions
    # Format: timestamp with 6 decimal places (microseconds)
    timestamp_str = f"{current_time:.6f}".replace(".", "_")

    # Build S3 path - include run_id in filename if available
    if run_id is not None:
        filename = f"{run_id}_{timestamp_str}.parquet"
    else:
        filename = f"{timestamp_str}.parquet"
    s3_path = f"s3://{s3_bucket}/{s3_prefix}/{table_name}/{filename}"

    log("export", f"Exporting {row_count} rows from {table_name} to {s3_path}")

    # Export to S3
    cursor.execute(
        f"""
        COPY (
            SELECT * FROM {table_name}
            {where_clause}
        ) TO '{s3_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        where_params,
    )

    # Update sync state
    cursor.execute(
        """
        INSERT INTO sync_state (table_name, last_synced_timestamp)
        VALUES (?, ?)
        ON CONFLICT (table_name)
        DO UPDATE SET last_synced_timestamp = EXCLUDED.last_synced_timestamp
        """,
        (table_name, current_time),
    )

    log("export", f"Successfully exported {row_count} rows to {s3_path}")
    return s3_path


def flush_all_to_s3(
    train_run: TrainRun,
    s3_bucket: str,
    cursor,
    s3_prefix: str = "staging",
) -> list[str]:
    """
    Flush all metric tables to S3

    Args:
        train_run: The training run context
        s3_bucket: S3 bucket name (required)
        cursor: Thread-safe DuckDB cursor
        s3_prefix: Prefix within the bucket (default: "staging")

    Returns:
        List of S3 paths that were created
    """
    # Ensure S3 credentials are configured
    ensure_s3_credentials(cursor)

    # Tables that have run_id column (filter by run_id)
    RUN_ID_TABLES = [MODEL_PARAMETER, TRAIN_STEP_METRIC, TRAIN_EPOCH_METRIC, TRAIN_STEPS_TABLE_NAME]

    # Tables without run_id column (filter by timestamp only)
    TIMESTAMP_ONLY_TABLES = [
        CHECKPOINT_SAMPLE_METRIC,
        MODELS_TABLE_NAME,
        RUNS_TABLE_NAME,
        CHECKPOINTS_TABLE_NAME,
        ARTIFACTS_TABLE_NAME,
        ARTIFACT_CHUNKS_TABLE_NAME,
    ]

    exported_paths = []

    # Export run-specific tables
    for table_name in RUN_ID_TABLES:
        s3_path = flush_table_to_s3(
            table_name=table_name,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            cursor=cursor,
            run_id=train_run.run_id,
        )
        if s3_path:
            exported_paths.append(s3_path)

    # Export timestamp-only tables (no run_id filtering)
    for table_name in TIMESTAMP_ONLY_TABLES:
        s3_path = flush_table_to_s3(
            table_name=table_name,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            cursor=cursor,
            run_id=None,
        )
        if s3_path:
            exported_paths.append(s3_path)

    return exported_paths


def ensure_s3_credentials(cursor):
    """Ensure S3 credentials are configured in DuckDB using AnalyticsConfig"""
    from lib.analytics_config import analytics_config

    config = analytics_config()
    if not config.is_s3_staging():
        raise ValueError("S3 staging not configured")

    s3 = config.staging.s3

    cursor.execute("INSTALL aws")
    cursor.execute("LOAD aws")

    # path-style: endpoint/bucket/key (MinIO default)
    # vhost-style: bucket.endpoint/key (AWS default, some providers require it)
    cursor.execute(f"SET s3_url_style='{s3.url_style}'")

    # Determine if we should use SSL based on endpoint
    use_ssl = not s3.endpoint.startswith("http://")
    cursor.execute(f"SET s3_use_ssl={str(use_ssl).lower()}")

    # Strip protocol from endpoint - DuckDB adds it based on USE_SSL
    endpoint = s3.endpoint.replace("https://", "").replace("http://", "")

    cursor.execute(
        f"""
        CREATE OR REPLACE SECRET (
            TYPE s3,
            PROVIDER config,
            KEY_ID '{s3.key}',
            SECRET '{s3.secret}',
            REGION '{s3.region}',
            ENDPOINT '{endpoint}',
            URL_STYLE '{s3.url_style}',
            USE_SSL {str(use_ssl).lower()}
        )
        """
    )


def export_periodic(
    train_run: TrainRun,
    interval_seconds: int = 300,
    s3_bucket: str = None,
):
    """
    Periodically export data to S3

    This can be called in a background thread or at regular intervals
    during training.

    Args:
        train_run: The training run context
        interval_seconds: How often to export (default: 300 = 5 minutes)
        s3_bucket: S3 bucket name (required)
    """
    import threading
    import time
    from lib.export import EXPORT_LOCK

    # Load config to get bucket and prefix
    from lib.analytics_config import analytics_config
    config = analytics_config()

    if not config.is_s3_staging():
        raise ValueError("S3 staging not configured in AnalyticsConfig")

    # Use config values
    if s3_bucket is None:
        s3_bucket = config.staging.bucket
    s3_prefix = config.staging.prefix

    def export_loop():
        # Create a cursor for this thread (thread-safe per DuckDB docs)
        # Connection objects are NOT thread-safe, but cursors are
        # Note: We access duck.CONN dynamically (not imported) to get the current value
        if duck.CONN is None:
            log("export", "Error: DuckDB connection not initialized")
            return

        cursor = duck.CONN.cursor()
        from lib.log import log_next_in

        while True:
            with EXPORT_LOCK:
                try:
                    paths = flush_all_to_s3(
                        train_run, s3_bucket, cursor, s3_prefix
                    )
                    if paths:
                        log_next_in(
                            "export",
                            f"Exported {len(paths)} files to S3",
                            interval_seconds,
                        )
                except Exception as e:
                    import traceback

                    log("export", f"Error during export: {e}\n{traceback.format_exc()}")

                try:
                    from lib.staging_filesystem import flush_all_to_checkpoint
                    from lib.paths import get_or_create_checkpoint_path

                    checkpoint_path = get_or_create_checkpoint_path(
                        train_run.train_config
                    )
                    flush_all_to_checkpoint(train_run, checkpoint_path, cursor)
                except Exception as e:
                    import traceback

                    log(
                        "export",
                        f"Error during checkpoint export: {e}\n{traceback.format_exc()}",
                    )

            time.sleep(interval_seconds)

    thread = threading.Thread(target=export_loop, daemon=True)
    thread.start()
    return thread

"""
Filesystem-based staging for parquet files.

Used when S3 is not available or desired (e.g., shared NFS/Lustre on HPC).
"""
from pathlib import Path
from typing import Optional
import time
import shutil
import duckdb
from lib.train_dataclasses import TrainRun
from lib.log import log


def flush_table_to_filesystem(
    table_name: str,
    staging_dir: Path,
    cursor,
    run_id: Optional[int] = None,
    sync_key_prefix: str = "",
) -> Optional[Path]:
    """
    Flush a single table's new data to filesystem as Parquet

    Args:
        table_name: Name of the table to export
        staging_dir: Directory to write parquet files
        cursor: Thread-safe DuckDB cursor
        run_id: Optional training run ID (filters by run_id if provided)

    Returns:
        Path to parquet file if data was exported, None if no new data
    """
    # Ensure we're using the local database
    cursor.execute("USE local")

    sync_key = f"{sync_key_prefix}{table_name}"

    # Get last synced timestamp
    result = cursor.execute(
        """
        SELECT COALESCE(MAX(last_synced_timestamp), 0) as last_synced
        FROM sync_state
        WHERE table_name = ?
        """,
        (sync_key,),
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

    # Build filesystem path - include run_id in filename if available
    table_dir = staging_dir / table_name
    table_dir.mkdir(parents=True, exist_ok=True)

    if run_id is not None:
        filename = f"{run_id}_{timestamp_str}.parquet"
    else:
        filename = f"{timestamp_str}.parquet"

    file_path = table_dir / filename

    log("export", f"Exporting {row_count} rows from {table_name} to {file_path}")

    # Export to filesystem
    cursor.execute(
        f"""
        COPY (
            SELECT * FROM {table_name}
            {where_clause}
        ) TO '{file_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
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
        (sync_key, current_time),
    )

    log("export", f"Successfully exported {row_count} rows to {file_path}")
    return file_path


def flush_all_to_filesystem(
    train_run: TrainRun,
    staging_dir: Path,
    cursor,
) -> list[Path]:
    """
    Flush all metric tables to filesystem

    Args:
        train_run: The training run context
        staging_dir: Directory to write parquet files
        cursor: Thread-safe DuckDB cursor

    Returns:
        List of filesystem paths that were created
    """
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
        file_path = flush_table_to_filesystem(
            table_name=table_name,
            staging_dir=staging_dir,
            cursor=cursor,
            run_id=train_run.run_id,
        )
        if file_path:
            exported_paths.append(file_path)

    # Export timestamp-only tables (no run_id filtering)
    for table_name in TIMESTAMP_ONLY_TABLES:
        file_path = flush_table_to_filesystem(
            table_name=table_name,
            staging_dir=staging_dir,
            cursor=cursor,
            run_id=None,
        )
        if file_path:
            exported_paths.append(file_path)

    return exported_paths


def flush_all_to_checkpoint(
    train_run: TrainRun,
    checkpoint_path: Path,
    cursor,
) -> list[Path]:
    """
    Flush all metric tables to the checkpoint directory as parquet files.

    Uses a separate sync key prefix so checkpoint export tracks progress
    independently from the primary staging export.
    """
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

    RUN_ID_TABLES = [MODEL_PARAMETER, TRAIN_STEP_METRIC, TRAIN_EPOCH_METRIC, TRAIN_STEPS_TABLE_NAME]
    TIMESTAMP_ONLY_TABLES = [
        CHECKPOINT_SAMPLE_METRIC,
        MODELS_TABLE_NAME,
        RUNS_TABLE_NAME,
        CHECKPOINTS_TABLE_NAME,
        ARTIFACTS_TABLE_NAME,
        ARTIFACT_CHUNKS_TABLE_NAME,
    ]

    analytics_dir = checkpoint_path / "analytics"
    exported_paths = []

    for table_name in RUN_ID_TABLES:
        file_path = flush_table_to_filesystem(
            table_name=table_name,
            staging_dir=analytics_dir,
            cursor=cursor,
            run_id=train_run.run_id,
            sync_key_prefix="ckpt_",
        )
        if file_path:
            exported_paths.append(file_path)

    for table_name in TIMESTAMP_ONLY_TABLES:
        file_path = flush_table_to_filesystem(
            table_name=table_name,
            staging_dir=analytics_dir,
            cursor=cursor,
            run_id=None,
            sync_key_prefix="ckpt_",
        )
        if file_path:
            exported_paths.append(file_path)

    return exported_paths


def export_periodic_filesystem(
    train_run: TrainRun,
    staging_dir: Path,
    interval_seconds: int = 300,
):
    """
    Periodically export data to filesystem

    This can be called in a background thread or at regular intervals
    during training.

    Args:
        train_run: The training run context
        staging_dir: Directory to write parquet files
        interval_seconds: How often to export (default: 300 = 5 minutes)
    """
    import threading
    import time
    import lib.render_duck as duck
    from lib.export import EXPORT_LOCK
    from lib.log import log_next_in

    def export_loop():
        # Create a cursor for this thread (thread-safe per DuckDB docs)
        # Note: We access duck.CONN dynamically (not imported) to get the current value
        if duck.CONN is None:
            log("export", "Error: DuckDB connection not initialized")
            return

        cursor = duck.CONN.cursor()

        while True:
            with EXPORT_LOCK:
                try:
                    paths = flush_all_to_filesystem(train_run, staging_dir, cursor)
                    if paths:
                        log_next_in(
                            "export",
                            f"Exported {len(paths)} files to filesystem",
                            interval_seconds,
                        )
                except Exception as e:
                    import traceback

                    log("export", f"Error during export: {e}\n{traceback.format_exc()}")

                try:
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

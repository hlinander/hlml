"""Recurring export of local EQP analytics through Feed.

The local DuckDB remains the training process's recorder and checkpoint source.
This module owns one Feed session per EQP run, streams rows added since its
own sync cursor, and advances that cursor only after an acknowledged flush.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from lib.analytics_config import FeedTarget
from lib.log import log, log_error, log_next_in
from lib.train_dataclasses import TrainRun


_CURSOR_EPOCH_COLUMN = "__feed_cursor_epoch"


class FeedExportError(RuntimeError):
    """The exporter could not establish acknowledged delivery."""


@dataclass(frozen=True)
class _TableSpec:
    name: str
    filter_column: str
    filter_value: int


@dataclass
class _PendingExport:
    table_name: str
    boundary: float
    row_count: int = 0


class FeedExporter:
    """Stateful bridge between one local EQP run and one Feed session."""

    def __init__(
        self,
        train_run: TrainRun,
        target: FeedTarget,
        cursor,
        *,
        run=None,
        run_factory: Optional[Callable[..., object]] = None,
        db_lock=None,
    ) -> None:
        self.train_run = train_run
        self.target = target
        if target.chunk_size <= 0:
            raise ValueError("FeedTarget.chunk_size must be positive")
        self._db_lock = db_lock or threading.RLock()
        self.cursor = cursor
        with self._db_lock:
            self.cursor.execute("USE local")
            self.model_id = self._resolve_model_id()
        self._lock = threading.Lock()
        self._pending: Optional[_PendingExport] = None
        self._failure: Optional[str] = None
        self._closed = False
        self.run = run or self._create_run(run_factory)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def reference(self) -> str:
        destination = self.target.destination or getattr(self.run, "feed", "")
        return f"feed://{destination}/{self.run.id}"

    def export_pending(self, *, finish: bool = False, cursor=None) -> int:
        """Publish new rows and acknowledge them before advancing sync cursors."""
        with self._lock:
            if cursor is not None:
                with self._db_lock:
                    self.cursor = cursor
                    self.cursor.execute("USE local")
            if self._closed:
                return 0
            if self._failure is not None:
                raise FeedExportError(self._failure)

            delivered_rows = self._settle_pending()
            cutoff = time.time()
            specs = self._table_specs()
            for spec in specs:
                delivered_rows += self._emit_table(spec, cutoff)

            # A fresh run may only have its run-metadata event. Flush it even if
            # none of the local metric tables contains rows yet.
            if delivered_rows == 0:
                self._flush_without_cursor()

            if finish:
                report = self.run.finish(self.target.flush_timeout_seconds)
                if not report.successful:
                    raise FeedExportError(
                        _report_error("Feed final flush failed", report)
                    )
                self._closed = True

            return delivered_rows

    def _settle_pending(self) -> int:
        if self._pending is None:
            return 0
        report = self.run.flush(self.target.flush_timeout_seconds)
        if not report.complete:
            raise FeedExportError(
                _report_error(
                    "Feed flush timed out; events remain pending with their "
                    "original wire identities",
                    report,
                )
            )
        if not report.successful:
            self._failure = _report_error(
                "Feed could not deliver an export batch; source rows retained", report
            )
            raise FeedExportError(self._failure)

        pending = self._pending
        with self._db_lock:
            self.cursor.execute(
                """
                INSERT INTO sync_state (table_name, last_synced_timestamp)
                VALUES (?, ?)
                ON CONFLICT (table_name)
                DO UPDATE SET last_synced_timestamp = EXCLUDED.last_synced_timestamp
                """,
                (self._sync_key(pending.table_name), pending.boundary),
            )
        self._pending = None
        return pending.row_count

    def _flush_without_cursor(self) -> None:
        report = self.run.flush(self.target.flush_timeout_seconds)
        if not report.complete:
            raise FeedExportError(
                _report_error("Feed metadata flush timed out", report)
            )
        if not report.successful:
            self._failure = _report_error(
                "Feed could not deliver run metadata", report
            )
            raise FeedExportError(self._failure)

    def _emit_table(self, spec: _TableSpec, cutoff: float) -> int:
        delivered_rows = 0
        while True:
            rows, columns, boundary = self._next_chunk(spec, cutoff)
            if not rows:
                break
            assert boundary is not None
            self._pending = _PendingExport(
                table_name=spec.name,
                boundary=boundary,
            )
            try:
                for values in rows:
                    row = dict(zip(columns, values))
                    self._emit_row(spec.name, row)
                    self._pending.row_count += 1
            except Exception as error:
                # A single source row can emit several events. Even the first
                # row may be partially admitted when a later event fails.
                self._failure = (
                    "Feed export stopped after a partial enqueue; the local "
                    f"cursor was retained: {error}"
                )
                raise
            delivered_rows += self._settle_pending()
        return delivered_rows

    def _next_chunk(self, spec: _TableSpec, cutoff: float):
        with self._db_lock:
            last_synced = self._last_synced(spec.name)
            base_sql = f"""
                SELECT *, EPOCH(timestamp) AS {_CURSOR_EPOCH_COLUMN}
                FROM {spec.name}
                WHERE {spec.filter_column} = ?
                  AND EPOCH(timestamp) > ?
                  AND EPOCH(timestamp) <= ?
            """
            preview = self.cursor.execute(
                f"{base_sql} ORDER BY timestamp LIMIT ?",
                (spec.filter_value, last_synced, cutoff, self.target.chunk_size),
            )
            columns = [column[0] for column in preview.description]
            rows = preview.fetchall()
            if not rows:
                return [], columns, None

            cursor_index = columns.index(_CURSOR_EPOCH_COLUMN)
            # Keep the watermark in DuckDB's numeric domain. Converting the
            # TIMESTAMPTZ to a Python datetime and back can round it slightly
            # below EPOCH(timestamp), causing the boundary row to repeat forever.
            boundary = max(float(row[cursor_index]) for row in rows)
            if len(rows) == self.target.chunk_size:
                # Include every row tied at the boundary timestamp. Advancing a
                # timestamp-only cursor after a partial tie would otherwise lose rows.
                complete_chunk = self.cursor.execute(
                    f"{base_sql} AND EPOCH(timestamp) <= ? ORDER BY timestamp",
                    (spec.filter_value, last_synced, cutoff, boundary),
                )
                columns = [column[0] for column in complete_chunk.description]
                rows = complete_chunk.fetchall()

            cursor_index = columns.index(_CURSOR_EPOCH_COLUMN)
            columns.pop(cursor_index)
            rows = [row[:cursor_index] + row[cursor_index + 1 :] for row in rows]
            return rows, columns, boundary

    def _emit_row(self, table_name: str, row: dict) -> None:
        from lib.render_duck import (
            CHECKPOINT_SAMPLE_METRIC,
            CHECKPOINTS_TABLE_NAME,
            MODEL_PARAMETER,
            TRAIN_EPOCH_METRIC,
            TRAIN_STEP_METRIC,
            TRAIN_STEPS_TABLE_NAME,
        )

        if table_name == MODEL_PARAMETER:
            self._log_record(
                "model_parameters",
                {
                    "model_id": row["model_id"],
                    "eqp_run_id": row["run_id"],
                    "name": row["name"],
                    "value_type": row["type"],
                    _typed_column(row, "value"): _typed_value(row, "value"),
                    "source_timestamp": _timestamp(row["timestamp"]),
                },
            )
            return

        if table_name == TRAIN_STEP_METRIC:
            value = _typed_value(row, "value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._emit_metric(
                    metric=row["name"],
                    value=float(value),
                    step=row["step"],
                    kind="metric",
                    context="train",
                    row=row,
                )
            else:
                self._emit_typed_metric(row, value)
            return

        if table_name == TRAIN_EPOCH_METRIC:
            context = f"{row['dataset_split']}:{row['dataset']}"
            self._emit_metric(
                metric=row["name"],
                value=float(row["mean"]),
                step=row["step"],
                kind="epoch",
                context=context,
                row=row,
            )
            self._log_record(
                "epoch_metrics",
                {
                    "model_id": row["model_id"],
                    "eqp_run_id": row["run_id"],
                    "epoch": row["epoch"],
                    "step": row["step"],
                    "metric": row["name"],
                    "dataset": row["dataset"],
                    "dataset_split": row["dataset_split"],
                    "mean": row["mean"],
                    "min": row["min"],
                    "max": row["max"],
                    "count": row["count"],
                    "source_timestamp": _timestamp(row["timestamp"]),
                },
            )
            return

        if table_name == CHECKPOINT_SAMPLE_METRIC:
            mean = _typed_value(row, "mean")
            per_sample = _typed_value(row, "value_per_sample")
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                self._emit_metric(
                    metric=row["name"],
                    value=float(mean),
                    step=row["step"],
                    kind="evaluation",
                    context=row["dataset"],
                    row=row,
                )
            sample_ids = row["sample_ids"] or []
            values = per_sample or []
            record = {
                "model_id": row["model_id"],
                "step": row["step"],
                "metric": row["name"],
                "dataset": row["dataset"],
                "value_type": row["type"],
                _typed_column(row, "mean"): mean,
                "sample_count": len(sample_ids),
                "source_timestamp": _timestamp(row["timestamp"]),
            }
            # Feed infers native array types from their values. Empty arrays
            # carry no element type, so their zero length is represented by
            # sample_count and the array columns are omitted for that row.
            if sample_ids:
                record["sample_ids"] = sample_ids
            if values:
                record[f"values_{row['type']}"] = values
            self._log_record(
                "evaluation_samples",
                record,
            )
            return

        if table_name == TRAIN_STEPS_TABLE_NAME:
            sample_ids = row["sample_ids"] or []
            record = {
                "model_id": row["model_id"],
                "eqp_run_id": row["run_id"],
                "step": row["step"],
                "dataset": row["dataset"],
                "sample_count": len(sample_ids),
                "source_timestamp": _timestamp(row["timestamp"]),
            }
            if sample_ids:
                record["sample_ids"] = sample_ids
            self._log_record(
                "train_steps",
                record,
            )
            return

        if table_name == CHECKPOINTS_TABLE_NAME:
            record = {
                "model_id": row["model_id"],
                "step": row["step"],
                "source_timestamp": _timestamp(row["timestamp"]),
            }
            if row["path"] is not None:
                record["path"] = row["path"]
            self._log_record(
                "checkpoints",
                record,
            )
            return

        raise FeedExportError(f"Unsupported Feed export table: {table_name}")

    def _emit_metric(
        self,
        *,
        metric: str,
        value: float,
        step: Optional[int],
        kind: str,
        context: Optional[str],
        row: dict,
    ) -> None:
        record = {
            "metric": metric,
            "value": value,
            "kind": kind,
            "model_id": row["model_id"],
            "eqp_run_id": row.get("run_id", self.train_run.run_id),
            "source_timestamp": _timestamp(row["timestamp"]),
        }
        if step is not None:
            record["step"] = step
        if context is not None:
            record["context"] = context
        self._log_record(
            "metric",
            record,
        )

    def _emit_typed_metric(self, row: dict, value) -> None:
        self._log_record(
            "metric_values",
            {
                "model_id": row["model_id"],
                "eqp_run_id": row["run_id"],
                "metric": row["name"],
                "step": row["step"],
                "value_type": row["type"],
                _typed_column(row, "value"): value,
                "source_timestamp": _timestamp(row["timestamp"]),
            },
        )

    def _log_record(self, stream_name: str, record: dict) -> None:
        accepted = self.run.log_wait(
            stream_name,
            record,
            timeout=self.target.enqueue_timeout_seconds,
        )
        if not accepted:
            raise FeedExportError(
                f"Feed did not accept {stream_name!r} within "
                f"{self.target.enqueue_timeout_seconds}s"
            )

    def _last_synced(self, table_name: str) -> float:
        with self._db_lock:
            result = self.cursor.execute(
                "SELECT last_synced_timestamp FROM sync_state WHERE table_name = ?",
                (self._sync_key(table_name),),
            ).fetchone()
            if result is not None:
                return float(result[0])

            # Local checkpoint persistence is not a Feed acknowledgement.
            # Without a Feed watermark, replay rather than silently skip data.
            return 0.0

    def _sync_key(self, table_name: str) -> str:
        return f"feed:{table_name}"

    def _resolve_model_id(self) -> int:
        with self._db_lock:
            row = self.cursor.execute(
                "SELECT model_id FROM runs WHERE id = ? ORDER BY timestamp DESC LIMIT 1",
                (self.train_run.run_id,),
            ).fetchone()
            if row is None:
                raise FeedExportError(
                    f"No local run metadata found for run_id={self.train_run.run_id}"
                )
            return int(row[0])

    def _create_run(self, run_factory: Optional[Callable[..., object]]):
        if run_factory is None:
            try:
                from feed import init as run_factory
            except ImportError as error:
                raise FeedExportError(
                    "FeedTarget requires the pinned feed-python dependency; run uv sync"
                ) from error

        serialized = self.train_run.serialize_human()
        config = {
            "eqp": {
                "project": self.train_run.project,
                "run_id": self.train_run.run_id,
                "model_id": self.model_id,
                "train_id": serialized["train_id"],
                "ensemble_id": serialized["ensemble_id"],
            },
            "train_config": serialized["train_config"],
            "train_eval": serialized["train_eval"],
            "epochs": serialized["epochs"],
            "git_rev": serialized["git_rev"],
            "slurm_jobid": serialized["slurm_jobid"],
        }
        try:
            from lib.feed_session import init

            return init(
                feed=self.target.destination,
                server_url=self.target.server_url,
                spool_dir=self.target.spool_dir,
                name=f"{self.train_run.project}-{self.model_id}",
                group=self.train_run.project,
                tags=["eqp"],
                config=config,
                metadata={"eqp_run_id": self.train_run.run_id, "model_id": self.model_id},
                enqueue_timeout_seconds=self.target.enqueue_timeout_seconds,
                flush_timeout_seconds=self.target.flush_timeout_seconds,
                run_factory=run_factory,
            )
        except Exception as error:
            message = f"FATAL: Feed analytics initialization failed: {error}"
            log_error("export", message)
            raise FeedExportError(message) from error

    def _table_specs(self) -> list[_TableSpec]:
        from lib.render_duck import (
            CHECKPOINT_SAMPLE_METRIC,
            CHECKPOINTS_TABLE_NAME,
            MODEL_PARAMETER,
            TRAIN_EPOCH_METRIC,
            TRAIN_STEP_METRIC,
            TRAIN_STEPS_TABLE_NAME,
        )

        return [
            _TableSpec(MODEL_PARAMETER, "run_id", self.train_run.run_id),
            _TableSpec(TRAIN_STEP_METRIC, "run_id", self.train_run.run_id),
            _TableSpec(TRAIN_EPOCH_METRIC, "run_id", self.train_run.run_id),
            _TableSpec(CHECKPOINT_SAMPLE_METRIC, "model_id", self.model_id),
            _TableSpec(TRAIN_STEPS_TABLE_NAME, "run_id", self.train_run.run_id),
            _TableSpec(CHECKPOINTS_TABLE_NAME, "model_id", self.model_id),
        ]


_EXPORTERS: dict[int, FeedExporter] = {}
_EXPORTERS_LOCK = threading.Lock()


def get_feed_exporter(train_run: TrainRun, target: FeedTarget, cursor) -> FeedExporter:
    import lib.render_duck as duck

    with _EXPORTERS_LOCK:
        exporter = _EXPORTERS.get(train_run.run_id)
        if exporter is None or exporter.closed:
            exporter = FeedExporter(
                train_run,
                target,
                cursor,
                db_lock=duck.CONN_LOCK,
            )
            _EXPORTERS[train_run.run_id] = exporter
        return exporter


def finish_feed_export(
    train_run: TrainRun, target: FeedTarget, cursor
) -> Optional[str]:
    exporter = get_feed_exporter(train_run, target, cursor)
    count = exporter.export_pending(finish=True, cursor=cursor)
    with _EXPORTERS_LOCK:
        _EXPORTERS.pop(train_run.run_id, None)
    log("export", f"Delivered {count} local analytics rows to {exporter.reference}")
    return exporter.reference


def export_periodic_feed(
    train_run: TrainRun,
    target: FeedTarget,
    interval_seconds: float,
):
    import lib.render_duck as duck

    def export_loop():
        if duck.CONN is None:
            log("export", "Error: DuckDB connection not initialized")
            return
        # Feed uses the existing connection under the same lock as training
        # writes. Duplicate connections can stall on EQP's attached in-memory
        # `local` database.
        cursor = duck.CONN
        exporter = None

        while exporter is None or not exporter.closed:
            try:
                with duck.CONN_LOCK:
                    _flush_checkpoint_analytics(train_run, cursor)
            except Exception as error:
                log("export", f"Error during checkpoint export: {error}")

            try:
                if exporter is None:
                    exporter = get_feed_exporter(train_run, target, cursor)
                count = exporter.export_pending(cursor=cursor)
                if count:
                    log_next_in(
                        "export",
                        f"Delivered {count} analytics rows to Feed",
                        interval_seconds,
                    )
            except Exception as error:
                log("export", f"Feed export pending: {error}")

            time.sleep(interval_seconds)

    thread = threading.Thread(target=export_loop, daemon=True)
    thread.start()
    return thread


def _flush_checkpoint_analytics(train_run: TrainRun, cursor) -> None:
    from lib.paths import get_or_create_checkpoint_path
    from lib.staging_filesystem import flush_all_to_checkpoint

    checkpoint_path = get_or_create_checkpoint_path(train_run.train_config)
    flush_all_to_checkpoint(train_run, checkpoint_path, cursor)


def _typed_value(row: dict, prefix: str):
    value_type = row["type"]
    return row[f"{prefix}_{value_type}"]


def _typed_column(row: dict, prefix: str) -> str:
    return f"{prefix}_{row['type']}"


def _timestamp(value) -> float:
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    return float(value)


def _report_error(prefix: str, report) -> str:
    message = (
        f"{prefix}: delivered={report.delivered}, filtered={report.filtered}, "
        f"failed={report.failed}, pending={report.pending}, "
        f"persisted_pending={report.persisted_pending}, unsaved={report.unsaved}, "
        f"spool_path={report.spool_path}"
    )
    if report.storage_error:
        message += f", storage_error={report.storage_error}"
    return message

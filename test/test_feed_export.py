from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

import lib.render_duck as duck
from lib.analytics_config import FeedTarget
from lib.export_feed import FeedExportError, FeedExporter

sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_train_run


@dataclass(frozen=True)
class _Report:
    delivered: int = 0
    filtered: int = 0
    dropped: int = 0
    pending: int = 0
    complete: bool = True
    persisted_pending: int = 0
    unsaved: int = 0
    spool_path: str = ""
    storage_error: str = ""

    @property
    def successful(self):
        return self.complete and self.dropped == 0 and not self.storage_error

    @property
    def failed(self):
        return self.dropped


class _FakeRun:
    id = "fake-session"

    def __init__(self, reports=None, feed=None):
        self.events = []
        self.reports = list(reports or [])
        self.finished = False
        self.feed = feed

    def emit_wait(self, stream_name, fields, *, timeout):
        self.events.append({
            "stream_name": stream_name,
            "data": {field.name: field.data_value() for field in fields},
            "schema": {field.name: field.type_descriptor() for field in fields},
            "timeout": timeout,
        })
        return True

    def log_wait(self, stream_name, record, *, timeout):
        self.events.append(
            {
                "stream_name": stream_name,
                "data": dict(record),
                "timeout": timeout,
            }
        )
        return True

    def flush(self, timeout):
        if self.reports:
            return self.reports.pop(0)
        return _Report()

    def finish(self, timeout):
        self.finished = True
        return _Report()


@pytest.fixture
def local_duck():
    if duck.CONN is not None:
        duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False
    duck.ensure_duck(None, True)
    yield
    if duck.CONN is not None:
        duck.CONN.close()
    duck.CONN = None
    duck.SCHEMA_ENSURED = False


def _local_run():
    train_run = create_train_run()
    model_id = duck.insert_model(train_run)
    duck.insert_run(train_run.run_id, model_id)
    return train_run, model_id


def test_train_runs_receive_distinct_default_ids():
    assert create_train_run().run_id != create_train_run().run_id


def test_default_feed_project_is_delegated_to_feed_init(local_duck):
    train_run, _model_id = _local_run()
    captured = {}

    def init_feed(**kwargs):
        import feed
        import inspect
        inspect.signature(feed.init).bind(**kwargs)
        captured.update(kwargs)
        return _FakeRun(feed="lab/paper")

    exporter = FeedExporter(
        train_run,
        FeedTarget(),
        duck.CONN,
        run_factory=init_feed,
    )

    assert captured["feed"] is None
    assert exporter.reference == "feed://lab/paper/fake-session"
    event = exporter.run.events[0]
    assert event['stream_name'] == 'runs'
    assert event['schema']['config'] == 'variant'
    assert event['data']['config']['eqp']['run_id'] == train_run.run_id
    assert event['data']['eqp_run_id'] == train_run.run_id
    assert event['data']['model_id'] == exporter.model_id
    assert event['data']['config']['train_config']['scheduler_config'] is None


def test_feed_init_failure_is_logged_loudly(local_duck, monkeypatch):
    train_run, _model_id = _local_run()
    logged = []
    monkeypatch.setattr(
        "lib.export_feed.log_error", lambda tag, message: logged.append((tag, message))
    )

    def ambiguous(**_kwargs):
        raise RuntimeError(
            "Feed project is ambiguous; run `feed use organization/project`"
        )

    with pytest.raises(FeedExportError, match="FATAL.*ambiguous"):
        FeedExporter(
            train_run,
            FeedTarget(),
            duck.CONN,
            run_factory=ambiguous,
        )

    assert logged == [
        (
            "export",
            "FATAL: Feed analytics initialization failed: Feed project is "
            "ambiguous; run `feed use organization/project`",
        )
    ]


def test_exports_local_tables_and_advances_dedicated_cursors(local_duck):
    train_run, model_id = _local_run()
    duck.insert_model_parameter(model_id, train_run.run_id, "learning_rate", 0.001)
    duck.insert_train_step_metric(model_id, train_run.run_id, "loss", 1, 0.5)
    duck.insert_train_epoch_metric(
        model_id,
        train_run.run_id,
        1,
        1,
        "loss",
        "TestDataset",
        "train",
        0.5,
        0.4,
        0.6,
        10,
    )
    duck.insert_checkpoint_sample_metric(
        model_id,
        1,
        "accuracy",
        "TestDataset",
        [10, 11],
        0.75,
        [1.0, 0.5],
    )
    duck.insert_train_step(model_id, train_run.run_id, 1, "TestDataset", [10, 11])
    duck.insert_checkpoint(model_id, 1, None)

    run = _FakeRun()
    exporter = FeedExporter(
        train_run,
        FeedTarget(project="org/project", chunk_size=2),
        duck.CONN,
        run=run,
    )

    assert exporter.export_pending() == 6
    names = [event["stream_name"] for event in run.events]
    assert "model_parameters" in names
    assert names.count("metric") == 3
    assert "epoch_metrics" in names
    assert "evaluation_samples" in names
    assert "train_steps" in names
    assert "checkpoints" in names

    metric = next(
        event
        for event in run.events
        if event["stream_name"] == "metric" and event["data"]["kind"] == "metric"
    )
    assert metric["data"]["model_id"] == model_id
    assert metric["data"]["eqp_run_id"] == train_run.run_id
    assert metric["data"]["metric"] == "loss"
    assert metric["data"]["value"] == 0.5

    parameter = next(
        event for event in run.events if event["stream_name"] == "model_parameters"
    )
    assert parameter["data"]["name"] == "learning_rate"
    assert parameter["data"]["value_type"] == "float"
    assert parameter["data"]["value_float"] == pytest.approx(0.001)

    sync_keys = {
        row[0]
        for row in duck.CONN.execute(
            "SELECT table_name FROM sync_state WHERE table_name LIKE 'feed:%'"
        ).fetchall()
    }
    assert sync_keys == {
        "feed:model_parameter",
        "feed:train_step_metric",
        "feed:train_epoch_metric",
        "feed:checkpoint_sample_metric",
        "feed:train_steps",
        "feed:checkpoints",
    }

    event_count = len(run.events)
    assert exporter.export_pending() == 0
    assert len(run.events) == event_count


def test_timeout_reuses_pending_events_instead_of_reemitting_rows(local_duck):
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, "loss", 1, 0.5)
    run = _FakeRun(
        reports=[
            _Report(pending=1, complete=False),
            _Report(delivered=1),
        ]
    )
    exporter = FeedExporter(
        train_run,
        FeedTarget(project="org/project"),
        duck.CONN,
        run=run,
    )

    with pytest.raises(FeedExportError, match="original wire identities"):
        exporter.export_pending()
    assert len(run.events) == 1
    assert (
        duck.CONN.execute(
            "SELECT COUNT(*) FROM sync_state WHERE table_name = 'feed:train_step_metric'"
        ).fetchone()[0]
        == 0
    )

    assert exporter.export_pending() == 1
    assert len(run.events) == 1
    assert (
        duck.CONN.execute(
            "SELECT COUNT(*) FROM sync_state WHERE table_name = 'feed:train_step_metric'"
        ).fetchone()[0]
        == 1
    )


def test_chunk_boundary_keeps_all_rows_with_the_same_timestamp(local_duck):
    train_run, model_id = _local_run()
    for step in range(3):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", step, 1.0 / (step + 1)
        )
    duck.CONN.execute(
        """
        UPDATE train_step_metric
        SET timestamp = TIMESTAMPTZ '2026-01-01 00:00:00+00'
        WHERE run_id = ?
        """,
        (train_run.run_id,),
    )

    run = _FakeRun()
    exporter = FeedExporter(
        train_run,
        FeedTarget(project="org/project", chunk_size=2),
        duck.CONN,
        run=run,
    )

    assert exporter.export_pending() == 3
    assert len(run.events) == 3
    assert exporter.export_pending() == 0
    assert len(run.events) == 3


def test_cursor_uses_duckdb_epoch_without_timestamp_rounding(local_duck):
    train_run, model_id = _local_run()
    duck.insert_checkpoint(model_id, 0, None)
    duck.CONN.execute(
        """
        UPDATE checkpoints
        SET timestamp = TIMESTAMPTZ '2026-08-24 18:03:15.755606+00'
        WHERE model_id = ?
        """,
        (model_id,),
    )

    exporter = FeedExporter(
        train_run,
        FeedTarget(project="org/project"),
        duck.CONN,
        run=_FakeRun(),
    )
    spec = next(spec for spec in exporter._table_specs() if spec.name == "checkpoints")

    rows, columns, boundary = exporter._next_chunk(spec, float("inf"))
    assert len(rows) == 1
    assert "__feed_cursor_epoch" not in columns
    python_epoch = rows[0][columns.index("timestamp")].timestamp()
    assert boundary > python_epoch

    duck.CONN.execute(
        """
        INSERT INTO sync_state (table_name, last_synced_timestamp)
        VALUES (?, ?)
        """,
        (exporter._sync_key(spec.name), boundary),
    )

    rows, _, boundary = exporter._next_chunk(spec, float("inf"))
    assert rows == []
    assert boundary is None


def test_feed_destination_alias_and_spool(local_duck, tmp_path):
    train_run, _ = _local_run()
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return _FakeRun(feed=kwargs['feed'])

    exporter = FeedExporter(train_run, FeedTarget(feed='lab/weather', spool_dir=str(tmp_path)),
                            duck.CONN, run_factory=factory)
    assert captured == dict(feed='lab/weather', server_url=None, spool_dir=str(tmp_path))
    assert exporter.reference == 'feed://lab/weather/fake-session'
    assert FeedTarget(project='lab/weather').destination == 'lab/weather'
    with pytest.raises(ValueError, match='same feed'):
        FeedTarget(feed='lab/weather', project='lab/other')


def test_checkpoint_watermark_is_not_delivery_acknowledgement(local_duck):
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 1, 0.5)
    duck.CONN.execute("INSERT INTO sync_state VALUES ('ckpt_train_step_metric', 9999999999)")
    run = _FakeRun()
    exporter = FeedExporter(train_run, FeedTarget(), duck.CONN, run=run)
    assert exporter.export_pending() == 1
    assert len(run.events) == 1
    assert duck.CONN.execute('SELECT count(*) FROM train_step_metric').fetchone()[0] == 1


def test_first_source_row_partial_enqueue_never_advances_cursor(local_duck):
    train_run, model_id = _local_run()
    duck.insert_train_epoch_metric(model_id, train_run.run_id, 1, 1, 'loss',
                                   'TestDataset', 'train', .5, .4, .6, 10)

    class PartialRun(_FakeRun):
        def log_wait(self, stream_name, record, *, timeout):
            if stream_name == 'epoch_metrics':
                return False
            return super().log_wait(stream_name, record, timeout=timeout)

    run = PartialRun()
    exporter = FeedExporter(train_run, FeedTarget(), duck.CONN, run=run)
    with pytest.raises(FeedExportError, match='did not accept'):
        exporter.export_pending()
    assert len(run.events) == 1
    with pytest.raises(FeedExportError, match='partial enqueue'):
        exporter.export_pending()
    assert duck.CONN.execute("SELECT count(*) FROM sync_state WHERE table_name LIKE 'feed:%'").fetchone()[0] == 0
    assert duck.CONN.execute('SELECT count(*) FROM train_epoch_metric').fetchone()[0] == 1


@pytest.mark.parametrize('report', [
    _Report(dropped=1),
    _Report(storage_error='disk full'),
    _Report(pending=1, persisted_pending=1, complete=False),
    _Report(pending=1, unsaved=1, complete=False),
])
def test_failed_or_unacknowledged_delivery_retains_local_rows(local_duck, report):
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 1, .5)
    exporter = FeedExporter(train_run, FeedTarget(), duck.CONN, run=_FakeRun([report]))
    with pytest.raises(FeedExportError):
        exporter.export_pending()
    assert duck.CONN.execute("SELECT count(*) FROM sync_state WHERE table_name LIKE 'feed:%'").fetchone()[0] == 0
    assert duck.CONN.execute('SELECT count(*) FROM train_step_metric').fetchone()[0] == 1


def test_recreated_exporter_uses_acknowledged_cursor_without_dropping_new_rows(local_duck):
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 1, .5)
    first = FeedExporter(train_run, FeedTarget(), duck.CONN, run=_FakeRun())
    assert first.export_pending() == 1
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 2, .25)
    run = _FakeRun()
    second = FeedExporter(train_run, FeedTarget(), duck.CONN, run=run)
    assert second.export_pending() == 1
    assert [event['data']['step'] for event in run.events] == [2]
    assert duck.CONN.execute('SELECT count(*) FROM train_step_metric').fetchone()[0] == 2


def test_final_export_saves_checkpoint_analytics_before_feed_failure(local_duck, monkeypatch):
    from lib.analytics_config import AnalyticsConfig
    from lib.export import export_all
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 1, .5)
    calls = []
    monkeypatch.setattr('lib.export_feed._flush_checkpoint_analytics',
                        lambda *_args: calls.append('persist'))

    def fail(*args, **kwargs):
        calls.append('feed')
        raise FeedExportError('offline')

    monkeypatch.setattr('lib.export_feed.finish_feed_export', fail)
    with pytest.raises(FeedExportError, match='offline'):
        export_all(train_run, AnalyticsConfig(staging=FeedTarget()))
    assert calls == ['persist', 'feed']
    assert duck.CONN.execute('SELECT count(*) FROM train_step_metric').fetchone()[0] == 1

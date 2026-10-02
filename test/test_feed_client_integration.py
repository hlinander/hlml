import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest
import requests

import lib.render_duck as duck
from lib.analytics_config import FeedTarget
from lib.export_feed import FeedExporter, FeedExportError
from test.test_feed_export import local_duck, _local_run


@pytest.fixture
def local_feed(tmp_path, monkeypatch):
    monkeypatch.setenv('FEED_API_KEY', 'local-test-only')
    monkeypatch.delenv('FEED_INGEST_URL', raising=False)
    monkeypatch.setenv('FEED_SPOOL_DIR', str(tmp_path / 'spool'))
    request = requests.Session.request

    def local_only(session, method, url, **kwargs):
        assert urlsplit(url).hostname == '127.0.0.1', 'Test attempted external traffic'
        return request(session, method, url, **kwargs)

    monkeypatch.setattr(requests.Session, 'request', local_only)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200, {'rules': []})

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            if self.headers.get('Content-Encoding') == 'gzip':
                body = gzip.decompress(body)
            batch = json.loads(body)
            self.server.batches.append(batch)
            count = len(batch['events'])
            self.respond(200, {'ingested': count if self.server.acknowledge else 0,
                               'dropped': 0})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.batches = []
    server.acknowledge = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}', server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def events(server):
    return [(batch['schemas'][item['schema_hash']]['$schema_name'], item['data'])
            for batch in server.batches for item in batch['events']]


def test_current_client_exports_linkable_metadata_and_metrics(local_duck, local_feed, tmp_path):
    url, server = local_feed
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 1, .5)
    target = FeedTarget(feed='tests/metrics', server_url=url, spool_dir=str(tmp_path / 'client-spool'))
    exporter = FeedExporter(train_run, target, duck.CONN)
    try:
        assert exporter.export_pending(finish=True) == 1
        received = dict(events(server))
        assert received['runs']['eqp_run_id'] == received['metric']['eqp_run_id'] == train_run.run_id
        assert received['runs']['model_id'] == received['metric']['model_id'] == model_id
        assert received['runs']['config']['train_config']['scheduler_config'] is None
        assert received['metric']['value'] == .5
    finally:
        exporter.run.finish(timeout=0)


def test_incomplete_ingestion_ack_does_not_advance_cursor(local_duck, local_feed, tmp_path):
    url, server = local_feed
    server.acknowledge = False
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 1, .5)
    target = FeedTarget(feed='tests/metrics', server_url=url,
                        spool_dir=str(tmp_path / 'client-spool'), flush_timeout_seconds=.2)
    exporter = FeedExporter(train_run, target, duck.CONN)
    try:
        with pytest.raises(FeedExportError, match='timed out'):
            exporter.export_pending()
        assert duck.CONN.execute("SELECT count(*) FROM sync_state WHERE table_name LIKE 'feed:%'").fetchone()[0] == 0
        assert duck.CONN.execute('SELECT count(*) FROM train_step_metric').fetchone()[0] == 1
        server.acknowledge = True
        target.flush_timeout_seconds = 10
        assert exporter.export_pending(finish=True) == 1
        identities = {item['session_sequence_num'] for batch in server.batches for item in batch['events']}
        assert len(identities) == 2  # runs plus metric, retries keep their identities.
    finally:
        exporter.run.finish(timeout=0)


def test_client_restart_recovers_retained_spool_without_losing_source(local_duck, local_feed, tmp_path):
    url, server = local_feed
    server.acknowledge = False
    train_run, model_id = _local_run()
    duck.insert_train_step_metric(model_id, train_run.run_id, 'loss', 1, .5)
    target = FeedTarget(feed='tests/metrics', server_url=url,
                        spool_dir=str(tmp_path / 'restart-spool'), flush_timeout_seconds=.2)
    old = FeedExporter(train_run, target, duck.CONN)
    try:
        with pytest.raises(FeedExportError):
            old.export_pending()
    finally:
        report = old.run.finish(timeout=.5)
    assert report.persisted_pending > 0
    assert report.unsaved == 0
    server.acknowledge = True
    target.flush_timeout_seconds = 10
    new = FeedExporter(train_run, target, duck.CONN)
    try:
        assert new.export_pending(finish=True) == 1
        assert any(batch['session_id'] == old.run.id for batch in server.batches)
        assert any(batch['session_id'] == new.run.id for batch in server.batches)
        assert duck.CONN.execute('SELECT count(*) FROM train_step_metric').fetchone()[0] == 1
        # Local replay in a new session is at least once, not an exactly-once claim.
        assert all(record['value'] == .5 for name, record in events(server) if name == 'metric')
    finally:
        new.run.finish(timeout=0)

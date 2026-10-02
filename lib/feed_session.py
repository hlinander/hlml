"""Application run metadata on top of Feed's event-only client API."""

from collections.abc import Mapping


def init(
    feed=None,
    *,
    name=None,
    config=None,
    tags=None,
    group=None,
    metadata: Mapping | None = None,
    enqueue_timeout_seconds=30.0,
    flush_timeout_seconds=30.0,
    run_factory=None,
    **client_options,
):
    from feed import EventBuilder, init as client_init

    client = (run_factory or client_init)(feed=feed, **client_options)
    try:
        builder = (
            EventBuilder()
            .add_string('name', name if name is not None else client.id)
            .add_optional_string('group', group)
            .add_string_array('tags', list(tags or []))
            .add_variant('config', config if config is not None else {})
        )
        for key, value in (metadata or {}).items():
            builder.add(key, value)
        if not client.emit_wait('runs', builder.build(), timeout=enqueue_timeout_seconds):
            raise RuntimeError('Feed did not accept run metadata')
        return client
    except BaseException:
        client.finish(timeout=flush_timeout_seconds)
        raise

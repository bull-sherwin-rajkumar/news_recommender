from concurrent.futures import ThreadPoolExecutor

import pytest

from news_recommender.storage import Filters, IndexNotReady, Store


def test_connections_are_reused_per_thread_and_can_be_closed(indexed):
    store, _, _ = indexed
    with store.connect() as first:
        pass
    with store.connect() as second:
        assert second is first

    def in_worker():
        with store.connect() as connection:
            assert connection is not first
        with store.connect() as again:
            assert again is connection
        store.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(in_worker).result()
    store.close()
    with store.connect() as reopened:
        assert reopened is not first
    store.close()


def test_ready_status_is_cached_but_external_changes_invalidate_it(indexed):
    store, _, _ = indexed
    statements = []
    with store.connect() as connection:
        connection.set_trace_callback(statements.append)
    store.ensure_ready()
    store.ensure_ready()
    assert sum("SELECT value FROM manifest" in sql for sql in statements) == 1
    writer = Store(store.path)
    writer.set_meta("status", "building")
    with pytest.raises(IndexNotReady):
        store.ensure_ready()
    writer.set_meta("status", "ready")
    store.ensure_ready()
    store.set_meta("status", "building")
    with pytest.raises(IndexNotReady):
        store.ensure_ready()
    writer.close()


def test_article_summaries_use_one_query_without_loading_bodies(indexed):
    store, _, _ = indexed
    chunks = store.lexical('"flood"', Filters(), 20)
    ids = list(dict.fromkeys(chunk.article_id for chunk in chunks))
    statements = []
    with store.connect() as connection:
        connection.set_trace_callback(statements.append)
    summaries = store.article_summaries(ids)
    assert len(statements) == 1
    assert " IN (" in statements[0]
    assert "body" not in statements[0]
    assert "NULL AS index_text" in statements[0]
    assert set(summaries) == set(ids)
    assert all(not hasattr(article, "body") for article in summaries.values())
    assert all(article.index_text is None for article in summaries.values())
    with_text = store.article_summaries(ids, include_index_text=True)
    assert all(article.index_text for article in with_text.values())
    assert store.article(ids[0]).body


def test_vector_search_caches_count_and_invalidates_after_writes(indexed, monkeypatch):
    _, vectors, _ = indexed
    original = vectors.count
    calls = []

    def count():
        calls.append(True)
        return original()

    monkeypatch.setattr(vectors, "count", count)
    for _ in range(3):
        assert vectors.search([1.0, 0.0], Filters(), 5)
    assert len(calls) == 1
    vectors.upsert([], [], {})
    vectors.search([1.0, 0.0], Filters(), 5)
    assert len(calls) == 2

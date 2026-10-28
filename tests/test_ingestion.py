from dataclasses import replace

import pytest

from news_recommender.config import DATASET, REVISION, Settings
from news_recommender.ingestion import (
    IngestionStats,
    dataset_rows,
    ingest,
    parse_article,
    source_url,
)
from news_recommender.nlp import normalize, split_chunks
from news_recommender.storage import Filters, IndexNotReady, Store, Vectors

from .conftest import source_row


def test_original_body_and_date_are_preserved(settings, nlp):
    body = "  A complete flood article.\n\nThe final paragraph is retained!  "
    ingest(settings, rows=[source_row(0, body=body)], nlp=nlp)
    store = Store(settings.database)
    hits = store.lexical('"flood"', Filters(), 5)
    article = store.article(hits[0].article_id)
    assert article.body == body
    assert article.index_text == normalize(body)
    assert article.published_day == 20210715
    assert article.published_at == "2021-07-15 02:46:59"
    assert store.counts()["chunks"] == 1


def test_missing_fields_and_bad_dates():
    stats = IngestionStats()
    assert parse_article({"Headline": "Missing body"}, stats) is None
    assert stats.invalid == 1
    row = source_row(0)
    row["Date published"] = "not-a-date"
    row["Url"] = "javascript:bad"
    article = parse_article(row, stats)
    assert article.published_at is None
    assert article.url is None
    assert stats.malformed_dates == stats.invalid_urls == 1
    row["Date published"] = None
    parse_article(row, stats)
    assert stats.missing_dates == 1


def test_deduplication_and_idempotence(settings, nlp):
    first = source_row(0)
    duplicate_body = source_row(1, body=first["Article text"])
    duplicate_url = source_row(2)
    duplicate_url["Url"] = first["Url"] + "#fragment"
    rows = [first, duplicate_body, duplicate_url]
    initial = ingest(settings, rows=rows, nlp=nlp)
    repeated = ingest(settings, rows=rows, nlp=nlp)
    assert initial["articles"] == repeated["articles"] == 1
    assert initial["vectors"] == repeated["vectors"]
    assert source_url("https://EXAMPLE.com/a#fragment") == "https://example.com/a"
    assert source_url("https://[invalid") is None


def test_limit_and_fingerprint_mismatch(settings, nlp):
    rows = [source_row(index) for index in range(5)]
    report = ingest(settings, limit=2, rows=rows, nlp=nlp)
    assert report["rows"] == report["articles"] == 2
    with pytest.raises(IndexNotReady, match="settings differ"):
        ingest(settings, rows=rows, nlp=nlp)
    with pytest.raises(ValueError, match="positive"):
        ingest(settings, limit=0, rows=rows, nlp=nlp)
    report = ingest(settings, rows=rows, nlp=nlp, rebuild=True)
    assert report["articles"] == 5


def test_interrupted_write_resumes(settings, nlp, monkeypatch):
    vectors = Vectors(settings, create=True)
    original = vectors.upsert

    def fail(*args):
        raise RuntimeError("interrupted vector write")

    monkeypatch.setattr(vectors, "upsert", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        ingest(settings, rows=[source_row(0)], nlp=nlp, vectors=vectors)
    store = Store(settings.database)
    with pytest.raises(IndexNotReady):
        store.ensure_ready()
    monkeypatch.setattr(vectors, "upsert", original)
    report = ingest(settings, rows=[source_row(0)], nlp=nlp, vectors=vectors)
    assert report["ready_articles"] == report["articles"] == report["vectors"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "Short story.",
        " ".join(f"token{index}" for index in range(1000)),
        "First paragraph.\n\n" * 100,
        "Flooding in caf\u00e9s: \u6d2a\u6c34 and rivers! " * 100,
    ],
)
def test_chunk_limits_coverage_and_offsets(text, nlp):
    text = normalize(text)
    chunks = split_chunks("parent", text, nlp.tokenizer, size=24, overlap=4)
    covered = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert len(nlp.tokenizer.encode(chunk.text, add_special_tokens=False)) <= 24
        covered.update(range(chunk.start, chunk.end))
    offsets = nlp.tokenizer(text)["offset_mapping"]
    assert all(all(pos in covered for pos in range(start, end)) for start, end in offsets)
    assert all(left.end > right.start for left, right in zip(chunks, chunks[1:]))
    assert chunks[-1].end == len(text)


def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        replace(settings, overlap_tokens=320)


def test_truncated_source_is_not_published(settings, nlp, monkeypatch):
    monkeypatch.setattr("news_recommender.ingestion.dataset_rows", lambda: iter([source_row(0)]))
    with pytest.raises(RuntimeError, match="Incomplete source"):
        ingest(settings, nlp=nlp)
    with pytest.raises(IndexNotReady):
        Store(settings.database).ensure_ready()


def test_dataset_uses_downloaded_cache_not_streaming(monkeypatch):
    calls = []
    rows = [source_row(0)]

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return rows

    monkeypatch.setattr("datasets.load_dataset", load)
    assert dataset_rows() is rows
    assert calls == [(DATASET, {"revision": REVISION, "split": "train", "streaming": False})]


def test_latency_configuration_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name in (
        "NEWS_CONTEXT_TOKENS", "NEWS_DEVICE", "NEWS_RERANKER_BACKEND", "NEWS_SEARCH_LIMIT"
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.device == "cuda"
    assert settings.reranker_backend == "onnx"
    assert settings.context_tokens == 64000
    assert settings.rerank_limit == 20
    assert settings.search_limit == 300
    monkeypatch.setenv("NEWS_DEVICE", "cpu")
    monkeypatch.setenv("NEWS_RERANKER_BACKEND", "torch")
    monkeypatch.setenv("NEWS_SEARCH_LIMIT", "500")
    overridden = Settings.from_env()
    assert overridden.device == "cpu" and overridden.reranker_backend == "torch"
    assert overridden.search_limit == 500
    monkeypatch.setenv("NEWS_SEARCH_LIMIT", "501")
    with pytest.raises(ValueError, match="Search limit"):
        Settings.from_env()

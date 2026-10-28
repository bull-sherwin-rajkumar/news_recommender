import argparse
import fcntl
import hashlib
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from urllib.parse import urlsplit, urlunsplit

from .config import DATASET, DATASET_ROWS, REVISION, Settings
from .nlp import NLP, normalize, split_chunks
from .storage import Article, Chunk, IndexNotReady, Store, Vectors

logger = logging.getLogger(__name__)


def field(row: Mapping[str, object], name: str) -> str:
    value = row.get(name)
    return "" if value is None else str(value).strip()


def source_url(value: str) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.query, ""))


@dataclass
class IngestionStats:
    rows: int = 0
    invalid: int = 0
    duplicates_or_existing: int = 0
    missing_dates: int = 0
    malformed_dates: int = 0
    invalid_urls: int = 0


def parse_article(row: Mapping[str, object], stats: IngestionStats) -> Article | None:
    body_value = row.get("Article text")
    headline = field(row, "Headline")
    if not isinstance(body_value, str) or not body_value.strip() or not headline:
        stats.invalid += 1
        return None
    text = normalize(body_value)
    if not text:
        stats.invalid += 1
        return None
    date = field(row, "Date published")
    published_at = None
    published_day = 0
    if date:
        try:
            parsed_date = datetime.fromisoformat(date.replace("Z", "+00:00"))
        except ValueError:
            stats.malformed_dates += 1
        else:
            published_at = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
            published_day = int(parsed_date.strftime("%Y%m%d"))
    else:
        stats.missing_dates += 1
    raw_url = field(row, "Url")
    url = source_url(raw_url)
    if raw_url and url is None:
        stats.invalid_urls += 1
    body_hash = hashlib.sha256(text.encode()).hexdigest()
    article_id = hashlib.sha256((url or body_hash).encode()).hexdigest()[:32]
    return Article(
        id=article_id,
        source_index=field(row, "Index"),
        url=url,
        headline=headline,
        description=field(row, "Description"),
        author=field(row, "Author"),
        published_at=published_at,
        published_day=published_day,
        category=field(row, "Category"),
        section=field(row, "Section"),
        keywords=field(row, "Keywords"),
        body=body_value,
        index_text=text,
        body_hash=body_hash,
    )


def dataset_rows() -> Iterable[Mapping[str, object]]:
    from datasets import load_dataset

    return load_dataset(DATASET, revision=REVISION, split="train", streaming=False)


def ingest(
    settings: Settings,
    limit: int | None = None,
    *,
    rebuild: bool = False,
    rows: Iterable[Mapping[str, object]] | None = None,
    nlp: NLP | None = None,
    vectors: Vectors | None = None,
) -> dict[str, object]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive; omit it for the full dataset.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with (settings.data_dir / ".ingest.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another ingestion is already running in this data directory."
            ) from exc
        return _ingest(settings, limit, rebuild, rows, nlp, vectors)


def _ingest(settings, limit, rebuild, rows, nlp, vectors) -> dict[str, object]:
    store = Store(settings.database, create=True)
    nlp = nlp or NLP(settings)
    dimension = nlp.encoder.get_sentence_embedding_dimension()
    fingerprint = settings.fingerprint(limit, dimension)
    previous = store.meta("fingerprint")
    if previous and previous != fingerprint and not rebuild:
        raise IndexNotReady(
            "Index settings differ. Use a different NEWS_DATA_DIR or run news-ingest --rebuild "
            "with the intended --limit (omitting --limit selects the full corpus)."
        )
    vectors = vectors or Vectors(settings, create=True)
    if rebuild:
        store.set_meta("status", "building")
        vectors.reset()
        store.reset()
    store.set_meta("fingerprint", fingerprint)
    store.set_meta("status", "building")
    stats = IngestionStats()
    batch: list[Chunk] = []
    articles: dict[str, Article] = {}
    write_queue: list[tuple[Article, list[Chunk]]] = []

    def flush() -> None:
        if not batch:
            return
        texts = [nlp.document(articles[c.article_id].headline, c.text) for c in batch]
        embeddings = nlp.embed_documents(texts)
        if any(len(embedding) != dimension for embedding in embeddings):
            raise ValueError("Embedding dimensions do not match the index manifest.")
        vectors.upsert(batch, embeddings, articles)
        store.mark_indexed(list(articles))
        batch.clear()
        articles.clear()

    def enqueue(article: Article, chunks: list[Chunk]) -> None:
        articles[article.id] = article
        batch.extend(chunks)
        if len(batch) >= settings.batch_size * 4:
            flush()

    def flush_writes() -> None:
        added = store.add_many(write_queue)
        stats.duplicates_or_existing += len(write_queue) - len(added)
        for article, chunks in added:
            enqueue(article, chunks)
        write_queue.clear()

    for article_id in store.pending():
        enqueue(store.pending_article(article_id), store.chunks_for(article_id))
    flush()

    source = rows if rows is not None else dataset_rows()
    for row in islice(source, limit):
        stats.rows += 1
        article = parse_article(row, stats)
        if article is None:
            continue
        chunks = split_chunks(
            article.id,
            article.index_text,
            nlp.tokenizer,
            settings.chunk_tokens,
            settings.overlap_tokens,
        )
        if not chunks:
            stats.invalid += 1
            continue
        write_queue.append((article, chunks))
        if len(write_queue) >= 64:
            flush_writes()
        if stats.rows % 1000 == 0:
            logger.info("Processed %s source rows; %s", stats.rows, store.counts())
    flush_writes()
    flush()
    expected_rows = min(limit or DATASET_ROWS, DATASET_ROWS)
    if rows is None and stats.rows != expected_rows:
        raise RuntimeError(
            f"Incomplete source: expected {expected_rows} rows, received {stats.rows}. "
            "The index remains unavailable; resume ingestion to retry."
        )
    counts = store.counts()
    vector_count = vectors.count()
    if (
        counts["articles"] == 0
        or counts["articles"] != counts["ready_articles"]
        or counts["chunks"] != counts["fts_chunks"]
        or counts["chunks"] != vector_count
    ):
        raise RuntimeError(f"Index integrity check failed: {counts}, vectors={vector_count}")
    report = {
        **vars(stats),
        **counts,
        "vectors": vector_count,
        "date_coverage": store.date_coverage(),
        "sample_limit": limit,
    }
    store.set_meta("report", report)
    store.set_meta("status", "ready")
    logger.info("Index ready: %s", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the CNN corpus locally.")
    parser.add_argument("--limit", type=int, help="Read only the first N source rows.")
    parser.add_argument("--rebuild", action="store_true", help="Replace the current local index.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest(Settings.from_env(), args.limit, rebuild=args.rebuild)


if __name__ == "__main__":
    main()

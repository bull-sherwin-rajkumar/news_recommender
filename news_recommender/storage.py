import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import Settings


class IndexNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class Article:
    id: str
    source_index: str
    url: str | None
    headline: str
    description: str
    author: str
    published_at: str | None
    published_day: int
    category: str
    section: str
    keywords: str
    body: str
    index_text: str
    body_hash: str


@dataclass(frozen=True)
class Chunk:
    id: str
    article_id: str
    ordinal: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Filters:
    category: str | None = None
    section: str | None = None
    date_from: int | None = None
    date_to: int | None = None
    article_id: str | None = None

    def __post_init__(self) -> None:
        for value in (self.date_from, self.date_to):
            if value is not None:
                datetime.strptime(str(value), "%Y%m%d")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("The start date must not be after the end date.")

    def sql(self) -> tuple[str, list[str | int]]:
        parts = ["a.indexed = 1"]
        values: list[str | int] = []
        for column, op, value in (
            ("category", "=", self.category),
            ("section", "=", self.section),
            ("published_day", ">=", self.date_from),
            ("published_day", "<=", self.date_to),
            ("id", "=", self.article_id),
        ):
            if value is not None:
                parts.append(f"a.{column} {op} ?")
                values.append(value)
        return " AND ".join(parts), values

    def vector(self) -> dict | None:
        parts = []
        for column, op, value in (
            ("category", "$eq", self.category),
            ("section", "$eq", self.section),
            ("published_day", "$gte", self.date_from),
            ("published_day", "$lte", self.date_to),
            ("article_id", "$eq", self.article_id),
        ):
            if value is not None:
                parts.append({column: {op: value}})
        return {"$and": parts} if len(parts) > 1 else parts[0] if parts else None


class Store:
    def __init__(self, path: Path, *, create: bool = False):
        self.path = path
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        elif not path.is_file():
            raise IndexNotReady("No index found. Run news-ingest --limit 1000 first.")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY, source_index TEXT NOT NULL, url TEXT UNIQUE,
                    headline TEXT NOT NULL, description TEXT NOT NULL, author TEXT NOT NULL,
                    published_at TEXT, published_day INTEGER NOT NULL,
                    category TEXT NOT NULL, section TEXT NOT NULL, keywords TEXT NOT NULL,
                    body TEXT NOT NULL, index_text TEXT NOT NULL, body_hash TEXT NOT NULL UNIQUE,
                    indexed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    article_id TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
                    text TEXT NOT NULL, UNIQUE(article_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS chunk_parent ON chunks(article_id);
                CREATE INDEX IF NOT EXISTS article_filters
                    ON articles(section, category, published_day);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    id UNINDEXED, article_id UNINDEXED, headline, keywords, text,
                    tokenize='porter unicode61 remove_diacritics 2'
                );
                CREATE TABLE IF NOT EXISTS manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    def reset(self) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM chunk_fts")
            db.execute("DELETE FROM articles")
            db.execute("DELETE FROM manifest")

    def set_meta(self, key: str, value: object) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO manifest VALUES (?, ?)", (key, json.dumps(value)))

    def meta(self, key: str) -> object:
        with self.connect() as db:
            row = db.execute("SELECT value FROM manifest WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def ensure_ready(self) -> None:
        if self.meta("status") != "ready":
            raise IndexNotReady(
                "Index is incomplete or ingestion is active. "
                "Resume news-ingest with the same options."
            )

    def add(self, article: Article, chunks: list[Chunk]) -> bool:
        return bool(self.add_many([(article, chunks)]))

    def add_many(
        self, items: list[tuple[Article, list[Chunk]]]
    ) -> list[tuple[Article, list[Chunk]]]:
        added = []
        with self.connect() as db:
            for article, chunks in items:
                duplicate = db.execute(
                    "SELECT id FROM articles WHERE id=? OR body_hash=? "
                    "OR (url IS NOT NULL AND url=?)",
                    (article.id, article.body_hash, article.url),
                ).fetchone()
                if duplicate:
                    continue
                values = asdict(article)
                columns = ", ".join(values)
                placeholders = ", ".join("?" for _ in values)
                db.execute(
                    f"INSERT INTO articles ({columns}) VALUES ({placeholders})",
                    list(values.values()),
                )
                db.executemany(
                    "INSERT INTO chunks VALUES (:id,:article_id,:ordinal,:start,:end,:text)",
                    [asdict(chunk) for chunk in chunks],
                )
                db.executemany(
                    "INSERT INTO chunk_fts VALUES (?,?,?,?,?)",
                    [
                        (c.id, article.id, article.headline, article.keywords, c.text)
                        for c in chunks
                    ],
                )
                added.append((article, chunks))
        return added

    def pending(self) -> list[str]:
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM articles WHERE indexed=0")]

    def mark_indexed(self, article_ids: list[str]) -> None:
        with self.connect() as db:
            db.executemany(
                "UPDATE articles SET indexed=1 WHERE id=?", [(value,) for value in article_ids]
            )

    def article(self, article_id: str) -> Article:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM articles WHERE id=? AND indexed=1", (article_id,)
            ).fetchone()
        if row is None:
            raise LookupError(f"Article {article_id} is not available in the ready index.")
        return self._article(row)

    def pending_article(self, article_id: str) -> Article:
        with self.connect() as db:
            row = db.execute("SELECT * FROM articles WHERE id=?", (article_id,)).fetchone()
        if row is None:
            raise LookupError(f"Unknown article: {article_id}")
        return self._article(row)

    @staticmethod
    def _article(row: sqlite3.Row) -> Article:
        values = dict(row)
        values.pop("indexed")
        return Article(**values)

    def chunks_for(self, article_id: str) -> list[Chunk]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM chunks WHERE article_id=? ORDER BY ordinal", (article_id,)
            ).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def chunks_by_ids(self, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        with self.connect() as db:
            rows = db.execute(
                f"SELECT c.* FROM chunks c JOIN articles a ON a.id=c.article_id "
                f"WHERE a.indexed=1 AND c.id IN ({','.join('?' for _ in ids)})",
                ids,
            ).fetchall()
        return {row["id"]: Chunk(**dict(row)) for row in rows}

    def lexical(self, expression: str, filters: Filters, limit: int | None) -> list[Chunk]:
        if not expression:
            return []
        where, values = filters.sql()
        with self.connect() as db:
            rows = db.execute(
                "SELECT c.* FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.id "
                "JOIN articles a ON a.id=c.article_id "
                f"WHERE chunk_fts MATCH ? AND {where} "
                "ORDER BY bm25(chunk_fts, 0, 0, 5, 3, 1), c.id"
                + (" LIMIT ?" if limit is not None else ""),
                [expression, *values, *([limit] if limit is not None else [])],
            ).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            return {
                "articles": db.execute("SELECT count(*) FROM articles").fetchone()[0],
                "ready_articles": db.execute(
                    "SELECT count(*) FROM articles WHERE indexed=1"
                ).fetchone()[0],
                "chunks": db.execute("SELECT count(*) FROM chunks").fetchone()[0],
                "fts_chunks": db.execute("SELECT count(*) FROM chunk_fts").fetchone()[0],
            }

    def facets(self) -> dict[str, list[str]]:
        with self.connect() as db:
            return {
                name: [
                    row[0]
                    for row in db.execute(
                        f"SELECT DISTINCT {name} FROM articles WHERE {name} != '' ORDER BY {name}"
                    )
                ]
                for name in ("category", "section")
            }

    def date_coverage(self) -> tuple[str | None, str | None]:
        with self.connect() as db:
            row = db.execute("SELECT min(published_at), max(published_at) FROM articles").fetchone()
        return row[0], row[1]


class Vectors:
    def __init__(self, settings: Settings, *, create: bool = False):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.client = chromadb.PersistentClient(
            path=str(settings.data_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = (
            self.client.get_or_create_collection(
                "news_chunks", metadata={"hnsw:space": "cosine"}, embedding_function=None
            )
            if create
            else self.client.get_collection("news_chunks", embedding_function=None)
        )

    def reset(self) -> None:
        self.client.delete_collection("news_chunks")
        self.collection = self.client.create_collection(
            "news_chunks", metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]], articles: dict[str, Article]
    ) -> None:
        maximum = self.client.get_max_batch_size()
        for start in range(0, len(chunks), maximum):
            part = chunks[start : start + maximum]
            self.collection.upsert(
                ids=[chunk.id for chunk in part],
                embeddings=embeddings[start : start + maximum],
                metadatas=[
                    {
                        "article_id": chunk.article_id,
                        "category": articles[chunk.article_id].category,
                        "section": articles[chunk.article_id].section,
                        "published_day": articles[chunk.article_id].published_day,
                    }
                    for chunk in part
                ],
            )

    def search(self, embedding: list[float], filters: Filters, limit: int) -> list[str]:
        count = self.count()
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(count, limit),
            where=filters.vector(),
            include=["distances"],
        )
        return result["ids"][0]

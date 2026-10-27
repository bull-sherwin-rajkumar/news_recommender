import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DATASET = "shivam9980/CNN-english-news"
REVISION = "463580eb67c121dd38ebd79055975d8e293dd4b4"
DATASET_ROWS = 37949


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ollama_model: str = "gemma4:12b"
    ollama_host: str = "http://localhost:11434"
    device: str = "auto"
    context_tokens: int = 64000
    ollama_timeout: float = 120
    chunk_tokens: int = 320
    overlap_tokens: int = 64
    batch_size: int = 128
    branch_limit: int = 50
    rerank_limit: int = 40

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_tokens < self.chunk_tokens <= 320:
            raise ValueError("Require 0 <= overlap_tokens < chunk_tokens <= 320.")
        if self.context_tokens < 4096 or self.ollama_timeout <= 0:
            raise ValueError("Use at least 4096 context tokens and a positive Ollama timeout.")
        if min(self.batch_size, self.branch_limit, self.rerank_limit) < 1:
            raise ValueError("Batch and retrieval limits must be positive.")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        return cls(
            data_dir=Path(os.getenv("NEWS_DATA_DIR", "data")).expanduser().resolve(),
            embedding_model=os.getenv("NEWS_EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("NEWS_RERANKER_MODEL", cls.reranker_model),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_host=os.getenv("OLLAMA_HOST", cls.ollama_host),
            device=os.getenv("NEWS_DEVICE", "auto"),
            context_tokens=int(os.getenv("NEWS_CONTEXT_TOKENS", "16384")),
            ollama_timeout=float(os.getenv("NEWS_OLLAMA_TIMEOUT", "120")),
        )

    @property
    def database(self) -> Path:
        return self.data_dir / "articles.sqlite3"

    def fingerprint(self, limit: int | None, dimension: int = 384) -> dict[str, object]:
        return {
            "schema": 1,
            "dataset": DATASET,
            "revision": REVISION,
            "limit": limit,
            "embedding_model": self.embedding_model,
            "embedding_dimension": dimension,
            "chunk_tokens": self.chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
            "normalization": "nfc-whitespace-v1",
        }

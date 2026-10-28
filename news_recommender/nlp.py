import logging
import re
import unicodedata
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Literal, Protocol

from .config import Settings
from .storage import Chunk

logger = logging.getLogger(__name__)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Tokenizer(Protocol):
    def __call__(self, text: str, **kwargs) -> dict: ...
    def encode(self, text: str, **kwargs) -> list[int]: ...
    def decode(self, ids: list[int], **kwargs) -> str: ...


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def is_gallery_boilerplate(text: str) -> bool:
    return text.count("Hide Caption") >= 2 or text.count("Photos:") >= 2


def split_chunks(
    article_id: str, text: str, tokenizer: Tokenizer, size: int = 320, overlap: int = 64
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be nonnegative and smaller than the chunk size.")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)[
        "offset_mapping"
    ]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        if end < len(offsets):
            for candidate in range(end, max(start + overlap + 1, end - 80), -1):
                char_end = offsets[candidate - 1][1]
                gap = text[char_end : offsets[candidate][0]]
                if "\n" in gap or text[char_end - 1 : char_end] in ".!?":
                    end = candidate
                    break
        left, right = offsets[start][0], offsets[end - 1][1]
        while len(tokenizer.encode(text[left:right], add_special_tokens=False)) > size:
            end -= 1
            if end <= start:
                raise ValueError(
                    "Tokenizer cannot represent a source span within the chunk budget."
                )
            right = offsets[end - 1][1]
        ordinal = len(chunks)
        chunks.append(
            Chunk(f"{article_id}:{ordinal}", article_id, ordinal, left, right, text[left:right])
        )
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return chunks


@lru_cache(maxsize=2)
def embedding_model(name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install CUDA-enabled PyTorch or set NEWS_DEVICE=cpu."
        )
    model = SentenceTransformer(
        name, device=None if device == "auto" else device, trust_remote_code=False
    )
    if model.device.type == "cuda":
        model.half()
    return model


@lru_cache(maxsize=2)
def reranking_model(name: str, device: str, backend: Literal["onnx", "torch"]):
    from sentence_transformers import CrossEncoder

    model_path = name
    if backend == "onnx" and not Path(name).is_dir():
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            name,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "onnx/model.onnx",
                "onnx/model.onnx_data",
            ],
        )
    model = CrossEncoder(
        model_path,
        device="cpu" if backend == "onnx" else None if device == "auto" else device,
        backend=backend,
        model_kwargs={"provider": "CPUExecutionProvider", "file_name": "onnx/model.onnx"}
        if backend == "onnx"
        else {},
        max_length=512,
    )
    if backend == "torch" and model.device.type == "cuda":
        model.model.half()
    return model


def clip_tokens(text: str, tokenizer: Tokenizer, limit: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= limit else tokenizer.decode(ids[:limit], skip_special_tokens=True)


class NLP:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def encoder(self):
        return embedding_model(self.settings.embedding_model, self.settings.device)

    @property
    def tokenizer(self) -> Tokenizer:
        return self.encoder.tokenizer

    @cached_property
    def ranker(self):
        return reranking_model(
            self.settings.reranker_model, self.settings.device, self.settings.reranker_backend
        )

    def warmup(self) -> None:
        self.embed_queries(["news"])
        self.rerank([("news", "A news report.")])

    def document(self, headline: str, chunk: str, tokenizer: Tokenizer | None = None) -> str:
        tokenizer = tokenizer or self.tokenizer
        return f"{clip_tokens(headline, tokenizer, 64)}\n{chunk}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        tokenized = self.encoder.tokenizer(texts, add_special_tokens=True, verbose=False)
        if any(len(ids) > self.encoder.max_seq_length for ids in tokenized["input_ids"]):
            raise ValueError("Embedding input exceeds its token budget; recheck chunk settings.")
        return self.encoder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
        ).tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        for query in queries:
            if len(self.tokenizer.encode(query, add_special_tokens=False)) > 96:
                raise ValueError("Search query is too long. Please use at most 96 tokens.")
        return self.embed_documents([QUERY_PREFIX + query for query in queries])

    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.ranker.predict(
            pairs, batch_size=self.settings.batch_size, show_progress_bar=False
        ).tolist()

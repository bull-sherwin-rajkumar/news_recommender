import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from .config import Settings
from .nlp import NLP, is_gallery_boilerplate, normalize
from .storage import Article, Chunk, Filters, IndexNotReady, Store, Vectors

STOP_WORDS = frozenset(
    "a an the and or of to in on for with about articles article news "
    "i me my want read find show please tell what which is are was were "
    "how does do did can could would should it this that some".split()
)

YEAR_PATTERN = r"(?<!\d)(?:19|20)\d{2}(?!\d)"


def required_terms(query: str) -> list[str]:
    return [
        *re.findall(YEAR_PATTERN, query),
        *[phrase for phrase in re.findall(r'"([^"]+)"', query) if phrase.strip()],
    ]


def matches_requirements(article: Article, query: str) -> bool:
    source = " ".join(
        f"{article.headline} {article.description} {article.index_text}".casefold().split()
    )
    return all(" ".join(term.casefold().split()) in source for term in required_terms(query))


def lexical_expression(query: str, *, match: Literal["any", "all", "phrase"] = "any") -> str:
    words = re.findall(r"[^\W_]+(?:['\u2019-][^\W_]+)*", normalize(query), flags=re.UNICODE)
    terms = list(
        dict.fromkeys(word.casefold() for word in words if word.casefold() not in STOP_WORDS)
    )
    phrases = [
        normalize(phrase)
        for phrase in re.findall(r'"([^"]+)"', query)
        if len(phrase) <= 160 and re.search(r"\w", phrase)
    ][:4]
    if match == "phrase":
        if not (terms or phrases) or len(words) > 64:
            return ""
        return '"' + " ".join(words).replace('"', '""') + '"'
    if match == "all":
        if len(terms) > 64:
            return ""
        operator = " AND "
    elif match == "any":
        operator = " OR "
    else:
        raise ValueError("Use lexical match any, all, or phrase.")
    return operator.join('"' + term.replace('"', '""') + '"' for term in [*phrases, *terms[:64]])


@dataclass(frozen=True)
class Hit:
    article: Article
    passages: list[Chunk]
    score: float

    @property
    def excerpt(self) -> str:
        return self.passages[0].text


def parent_ranks(chunks: list[Chunk], limit: int) -> list[tuple[str, list[Chunk]]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.article_id not in grouped and len(grouped) >= limit:
            continue
        group = grouped.setdefault(chunk.article_id, [])
        if len(group) < 2 and chunk.id not in {item.id for item in group}:
            group.append(chunk)
    return list(grouped.items())


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: Store | None = None,
        vectors: Vectors | None = None,
        nlp: NLP | None = None,
    ):
        self.settings = settings
        self.store = store or Store(settings.database)
        self.store.ensure_ready()
        self.vectors = vectors or Vectors(settings)
        self.nlp = nlp or NLP(settings)
        fingerprint = self.store.meta("fingerprint")
        if not isinstance(fingerprint, dict):
            raise IndexNotReady("Missing index manifest. Rebuild the index.")
        expected = settings.fingerprint(
            fingerprint.get("limit"), fingerprint.get("embedding_dimension", 384)
        )
        if fingerprint != expected:
            raise IndexNotReady("Embedding/chunk settings differ from the index. Rebuild it.")

    def search(
        self,
        queries: list[str],
        filters: Filters | None = None,
        *,
        limit: int = 5,
        method: str = "hybrid",
        prefer_lexical: bool = False,
    ) -> list[Hit]:
        self.store.ensure_ready()
        if not queries or not any(query.strip() for query in queries):
            raise ValueError("Enter a topic or question.")
        if not 1 <= limit <= 5 or method not in ("hybrid", "bm25"):
            raise ValueError("Use a result limit from 1 to 5 and method hybrid or bm25.")
        queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))[:2]
        filters = filters or Filters()
        fused: dict[str, float] = defaultdict(float)
        evidence: dict[str, dict[str, Chunk]] = defaultdict(dict)
        primary: dict[str, dict[int, str]] = defaultdict(dict)
        lexical_tiers: dict[str, int] = {}
        embeddings = self.nlp.embed_queries(queries) if method == "hybrid" else []
        for index, query in enumerate(queries):
            if prefer_lexical and index == 0:
                lexical = []
                expressions: set[str] = set()
                modes: tuple[Literal["phrase", "all", "any"], ...] = ("phrase", "all", "any")
                for tier, match in enumerate(modes):
                    expression = lexical_expression(query, match=match)
                    if not expression or expression in expressions:
                        continue
                    expressions.add(expression)
                    chunks = self.store.lexical(
                        expression, filters, self.settings.branch_limit * 10
                    )
                    chunks = [chunk for chunk in chunks if not is_gallery_boilerplate(chunk.text)]
                    for article_id, passages in parent_ranks(chunks, self.settings.branch_limit):
                        lexical_tiers.setdefault(article_id, tier)
                        lexical.extend(passages)
            else:
                lexical = self.store.lexical(
                    lexical_expression(query), filters, self.settings.branch_limit * 10
                )
            branches = [lexical]
            if method == "hybrid":
                ids = self.vectors.search(
                    embeddings[index], filters, self.settings.branch_limit * 10
                )
                chunks = self.store.chunks_by_ids(ids)
                branches.append([chunks[chunk_id] for chunk_id in ids if chunk_id in chunks])
            for branch_index, branch in enumerate(branches):
                branch = [chunk for chunk in branch if not is_gallery_boilerplate(chunk.text)]
                for rank, (article_id, chunks) in enumerate(
                    parent_ranks(branch, self.settings.branch_limit), 1
                ):
                    fused[article_id] += (1.0 if index == 0 else 0.5) / (60 + rank)
                    primary[article_id].setdefault(branch_index, chunks[0].id)
                    for chunk in chunks:
                        evidence[article_id].setdefault(chunk.id, chunk)
        candidates = [
            aid
            for aid in sorted(fused, key=lambda aid: (lexical_tiers.get(aid, 2), -fused[aid], aid))
            if matches_requirements(self.store.article(aid), queries[0])
        ][: self.settings.rerank_limit]
        if not candidates:
            return []
        if method == "bm25":
            return [
                Hit(self.store.article(aid), list(evidence[aid].values())[:2], fused[aid])
                for aid in candidates[:limit]
            ]
        passages = []
        for article_id in candidates:
            chunk_ids = dict.fromkeys([*primary[article_id].values(), *evidence[article_id]])
            passages.extend(evidence[article_id][chunk_id] for chunk_id in list(chunk_ids)[:2])
        pairs = [
            (
                queries[0],
                self.nlp.document(
                    self.store.article(chunk.article_id).headline,
                    chunk.text,
                    self.nlp.ranker.tokenizer,
                ),
            )
            for chunk in passages
        ]
        scores = self.nlp.rerank(pairs)
        scored: dict[str, list[tuple[float, Chunk]]] = defaultdict(list)
        for chunk, score in zip(passages, scores, strict=True):
            if not math.isfinite(score):
                raise RuntimeError("The reranker returned a non-finite relevance score.")
            scored[chunk.article_id].append((score, chunk))
        hits = []
        for article_id, items in scored.items():
            items.sort(key=lambda pair: (-pair[0], pair[1].id))
            hits.append(
                Hit(self.store.article(article_id), [item[1] for item in items], items[0][0])
            )
        hits.sort(key=lambda hit: (-hit.score, hit.article.id))
        selected: list[Hit] = []
        while hits and len(selected) < limit:
            # Keep lexical tiers ahead of supplements, even after diversity adjustment.
            best = max(
                hits,
                key=lambda hit: (
                    -lexical_tiers.get(hit.article.id, 2),
                    hit.score
                    - 0.5 * max((self._overlap(hit, other) for other in selected), default=0),
                ),
            )
            selected.append(best)
            hits.remove(best)
        return selected

    @staticmethod
    def _overlap(left: Hit, right: Hit) -> float:
        first = set(left.excerpt.casefold().split())
        second = set(right.excerpt.casefold().split())
        return len(first & second) / max(1, len(first | second))

from dataclasses import replace

import pytest

from news_recommender.ingestion import ingest
from news_recommender.nlp import is_gallery_boilerplate
from news_recommender.retrieval import Retriever, lexical_expression, parent_ranks
from news_recommender.storage import Filters, IndexNotReady, Store, Vectors

from .conftest import source_row


@pytest.fixture(params=[("fixturealpha", "fixturebeta"), ("fixturegamma", "fixturedelta")])
def lexical_corpus(settings, nlp, monkeypatch, request):
    first, second = request.param
    bodies = [
        f"{first} {second}. Phrase fixture.",
        f"{first} separated words {second}. Complete fixture.",
        f"{first}. Partial fixture.",
        "Supplemental candidate with unrelated wording.",
    ]
    rows = []
    for index, body in enumerate(bodies):
        row = source_row(index, body=body, section="primary" if index == 0 else "secondary")
        row.update(Headline=f"Fixture {index}", Keywords="", Description="")
        rows.append(row)
    vectors = Vectors(settings, create=True)
    ingest(settings, rows=rows, nlp=nlp, vectors=vectors)
    store = Store(settings.database)
    scores = {"Fixture 0": -100.0, "Fixture 1": -50.0, "Fixture 2": 50.0, "Fixture 3": 100.0}
    monkeypatch.setattr(
        nlp, "rerank", lambda pairs: [scores[text.split("\n")[0]] for _, text in pairs]
    )
    return Retriever(settings, store, vectors, nlp), f"{first} {second}"


def test_lexical_tiers_cannot_be_displaced_by_semantic_scores(lexical_corpus):
    retriever, query = lexical_corpus
    hits = retriever.search([query], prefer_lexical=True)
    assert [hit.article.source_index for hit in hits] == ["0", "1", "3", "2"]
    assert [hit.score for hit in hits] == [-100.0, -50.0, 100.0, 50.0]
    assert len({hit.article.id for hit in hits}) == 4
    assert [
        hit.article.source_index for hit in retriever.search([query], limit=2, prefer_lexical=True)
    ] == ["0", "1"]
    assert [hit.article.source_index for hit in retriever.search([query])] == ["3", "2", "1", "0"]


def test_phrase_survives_branch_and_rerank_budgets(lexical_corpus, monkeypatch):
    retriever, query = lexical_corpus
    retriever.settings = replace(retriever.settings, branch_limit=1, rerank_limit=1)
    original_lexical = retriever.store.lexical

    def lexical(expression, filters, limit):
        chunks = original_lexical(expression, filters, limit)
        if expression == lexical_expression(query):
            return [
                chunk
                for chunk in chunks
                if retriever.store.article(chunk.article_id).source_index == "2"
            ]
        return chunks

    monkeypatch.setattr(retriever.store, "lexical", lexical)
    unrelated = retriever.store.lexical('"unrelated"', Filters(), 1)[0]
    monkeypatch.setattr(retriever.vectors, "search", lambda *args: [unrelated.id])
    hits = retriever.search([query], prefer_lexical=True)
    assert [hit.article.source_index for hit in hits] == ["0"]


def test_reranking_stays_within_lexical_tiers(lexical_corpus):
    retriever, query = lexical_corpus
    hits = retriever.search([query.split()[0]], prefer_lexical=True)
    assert [hit.article.source_index for hit in hits] == ["2", "1", "0", "3"]


def test_all_term_matches_are_protected_when_no_phrase_matches(lexical_corpus):
    retriever, query = lexical_corpus
    reversed_query = " ".join(reversed(query.split()))
    hits = retriever.search([reversed_query], prefer_lexical=True)
    assert [hit.article.source_index for hit in hits] == ["1", "0", "3", "2"]
    retriever.settings = replace(retriever.settings, rerank_limit=1)
    shortlisted = retriever.search([reversed_query], prefer_lexical=True)
    assert len(shortlisted) == 1
    assert shortlisted[0].article.source_index in ("0", "1")


def test_stopword_only_query_uses_supplemental_ranking(lexical_corpus):
    retriever, _ = lexical_corpus
    hits = retriever.search(["I want to read articles about"], prefer_lexical=True)
    assert [hit.article.source_index for hit in hits] == ["3", "2", "1", "0"]


def test_lexical_priority_respects_filters_and_explicit_requirements(lexical_corpus):
    retriever, query = lexical_corpus
    hits = retriever.search([query], Filters(section="secondary"), prefer_lexical=True)
    assert hits[0].article.source_index == "1"
    assert all(hit.article.section == "secondary" for hit in hits)
    assert retriever.search([query], Filters(date_from=20220101), prefer_lexical=True) == []
    assert retriever.search([f"{query} 2099"], prefer_lexical=True) == []
    quoted = retriever.search([f'"{query}"'], prefer_lexical=True)
    assert [hit.article.source_index for hit in quoted] == ["0"]


def test_lexical_expression_modes_preserve_phrase_and_all_terms():
    query = "FixtureAlpha in the FixtureBeta"
    assert lexical_expression(query, match="phrase") == '"FixtureAlpha in the FixtureBeta"'
    assert lexical_expression(query, match="all") == '"fixturealpha" AND "fixturebeta"'
    assert lexical_expression(query) == '"fixturealpha" OR "fixturebeta"'
    assert lexical_expression("fixturealpha fixturealpha", match="all") == '"fixturealpha"'
    for match in ("phrase", "all"):
        assert lexical_expression("I want to read articles about", match=match) == ""
        assert lexical_expression(" ".join(f"fixture{i}" for i in range(65)), match=match) == ""


def test_all_lexical_modes_escape_fts_syntax(settings, indexed):
    store, _, _ = indexed
    for query in ['" OR * NEAR(foo) --', "'", "NOT (", "foo:bar", '"fixturealpha"']:
        for match in ("phrase", "all", "any"):
            store.lexical(lexical_expression(query, match=match), Filters(), 5)


def test_five_unique_articles_and_result_limit(settings, indexed):
    store, vectors, nlp = indexed
    retriever = Retriever(settings, store, vectors, nlp)
    hits = retriever.search(["coastal flood"])
    assert len(hits) == len({hit.article.id for hit in hits}) == 5
    assert len(retriever.search(["flood"], limit=2)) == 2


@pytest.mark.parametrize("score", [-100.0, -1.0, 0.0, 1.0, 10.0])
def test_finite_scores_do_not_gate_vector_candidates(settings, indexed, monkeypatch, score):
    store, vectors, nlp = indexed
    monkeypatch.setattr(store, "lexical", lambda *args: [])
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [score] * len(pairs))
    retriever = Retriever(settings, store, vectors, nlp)
    for limit in (1, 3, 5):
        hits = retriever.search(["fixture query"], limit=limit)
        assert len(hits) == len({hit.article.id for hit in hits}) == limit
        assert all(hit.score == score for hit in hits)


def test_score_order_and_best_passage_are_preserved(settings, indexed, monkeypatch):
    store, vectors, nlp = indexed
    with store.connect() as db:
        ids = [row[0] for row in db.execute("SELECT id FROM chunks ORDER BY id LIMIT 3")]
    chunks = list(store.chunks_by_ids(ids).values())
    extra = replace(chunks[0], id=chunks[0].id + "-extra", text="Additional fixture passage.")
    passages = [chunks[0], extra, *chunks[1:]]
    scores = {
        nlp.document(store.article(chunk.article_id).headline, chunk.text): float(index * 2 - 20)
        for index, chunk in enumerate(passages)
    }
    monkeypatch.setattr(store, "lexical", lambda *args: [*passages, chunks[0]])
    monkeypatch.setattr(vectors, "search", lambda *args: [])
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [scores[text] for _, text in pairs])
    retriever = Retriever(settings, store, vectors, nlp)
    hits = retriever.search(["fixture query"])
    assert [hit.score for hit in hits] == [-14.0, -16.0, -18.0]
    assert len({hit.article.id for hit in hits}) == 3
    assert hits[-1].passages == [extra, chunks[0]]


def test_diversity_adjusts_ranking_without_discarding_low_scores(settings, indexed, monkeypatch):
    store, vectors, nlp = indexed
    with store.connect() as db:
        ids = [row[0] for row in db.execute("SELECT id FROM chunks ORDER BY id LIMIT 3")]
    chunks = list(store.chunks_by_ids(ids).values())
    passages = [
        replace(chunk, text=text)
        for chunk, text in zip(
            chunks, ["shared words", "shared words", "different text"], strict=True
        )
    ]
    monkeypatch.setattr(store, "lexical", lambda *args: passages)
    monkeypatch.setattr(vectors, "search", lambda *args: [])
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [-1.0, -1.1, -1.2])
    hits = Retriever(settings, store, vectors, nlp).search(["fixture query"])
    assert [hit.article.id for hit in hits] == [passages[index].article_id for index in (0, 2, 1)]
    assert [hit.score for hit in hits] == [-1.0, -1.2, -1.1]


def test_empty_candidate_pool_returns_no_hits(settings, indexed, monkeypatch):
    store, vectors, nlp = indexed
    monkeypatch.setattr(store, "lexical", lambda *args: [])
    monkeypatch.setattr(vectors, "search", lambda *args: [])

    def unexpected_rerank(pairs):
        pytest.fail("An empty candidate pool must not be reranked.")

    monkeypatch.setattr(nlp, "rerank", unexpected_rerank)
    assert Retriever(settings, store, vectors, nlp).search(["fixture query"]) == []


def test_filters_apply_to_both_indexes(settings, nlp, monkeypatch):
    rows = [
        source_row(0, section="europe"),
        source_row(1, section="asia", category="business"),
    ]
    ingest(settings, rows=rows, nlp=nlp)
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [-100.0] * len(pairs))
    retriever = Retriever(settings, nlp=nlp)
    filters = Filters(section="asia", category="business", date_from=20210101, date_to=20211231)
    hits = retriever.search(["flood"], filters)
    assert len(hits) == 1
    assert hits[0].article.section == "asia"
    assert retriever.search(["flood"], Filters(date_from=20220101)) == []
    assert len(retriever.search(["flood"], Filters(article_id=hits[0].article.id))) == 1
    with pytest.raises(ValueError, match="start date"):
        Filters(date_from=20220101, date_to=20210101)


def test_safe_fts_and_stemming(settings, nlp):
    row = source_row(0, body="Coastal residents were evacuating after floods.")
    ingest(settings, rows=[row], nlp=nlp)
    retriever = Retriever(settings, nlp=nlp)
    assert retriever.store.lexical(lexical_expression("evacuation"), Filters(), 5)
    for query in ['" OR * NEAR(foo) --', "'", "NOT (", "foo:bar", '"flood"']:
        expression = lexical_expression(query)
        retriever.store.lexical(expression, Filters(), 5)
    assert lexical_expression("I want to read articles about") == ""
    assert '"New York"' in lexical_expression('news about "New York"')
    with pytest.raises(ValueError):
        Filters(date_from=20210230)


def test_parent_collapse_and_semantic_evidence(settings, nlp, monkeypatch):
    row = source_row(0, body=" ".join(["ordinary"] * 700 + ["needle"] * 30))
    row["Headline"] = "A report"
    ingest(settings, rows=[row], nlp=nlp)
    retriever = Retriever(settings, nlp=nlp)
    article_id = retriever.store.lexical('"needle"', Filters(), 1)[0].article_id
    chunks = retriever.store.chunks_for(article_id)
    assert len(parent_ranks(chunks, 50)) == 1
    monkeypatch.setattr(retriever.store, "lexical", lambda *args: chunks[:2])
    monkeypatch.setattr(retriever.vectors, "search", lambda *args: [chunks[-1].id])
    hits = retriever.search(["needle"])
    assert hits and "needle" in hits[0].excerpt


def test_changed_model_and_unready_index_rejected(settings, indexed):
    store, vectors, nlp = indexed
    with pytest.raises(IndexNotReady, match="settings differ"):
        Retriever(replace(settings, embedding_model="different"), store, vectors, nlp)
    retriever = Retriever(settings, store, vectors, nlp)
    store.set_meta("status", "building")
    with pytest.raises(IndexNotReady):
        retriever.search(["flood"])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_bm25_baseline_and_finite_scores(settings, indexed, monkeypatch, score):
    store, vectors, nlp = indexed
    retriever = Retriever(settings, store, vectors, nlp)
    assert len(retriever.search(["flood"], method="bm25")) == 5
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [score] * len(pairs))
    with pytest.raises(RuntimeError, match="non-finite"):
        retriever.search(["flood"])


def test_explicit_years_and_phrases_are_hard_requirements(settings, indexed, monkeypatch):
    store, vectors, nlp = indexed
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [-100.0] * len(pairs))
    retriever = Retriever(settings, store, vectors, nlp)
    assert retriever.search(["flood 2026"]) == []
    assert retriever.search(['flood "nonexistent phrase"']) == []


def test_repeated_gallery_is_not_article_evidence(settings, nlp, monkeypatch):
    row = source_row(0, body="Photos: gallery Flood damage. Hide Caption 1 of 175. " * 10)
    ingest(settings, rows=[row], nlp=nlp)
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [-100.0] * len(pairs))
    retriever = Retriever(settings, nlp=nlp)
    assert is_gallery_boilerplate(row["Article text"])
    assert retriever.search(["flood"]) == []

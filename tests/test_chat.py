import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from news_recommender.chat import NO_EVIDENCE, Chat, ChatError, PreparedQuery
from news_recommender.ingestion import ingest
from news_recommender.retrieval import Retriever
from news_recommender.storage import Filters

from .conftest import source_row

PLAN = {
    "standalone_question": "What happened during coastal flooding?",
    "topic_query": "coastal flood",
    "alternatives": [],
}


class Client:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if kwargs.get("stream"):
            def fragments():
                for part in response:
                    if isinstance(part, Exception):
                        raise part
                    yield SimpleNamespace(
                        message=SimpleNamespace(content=part), done=False
                    )
                yield SimpleNamespace(
                    message=SimpleNamespace(content=""), done=True, done_reason="stop"
                )
            return fragments()
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))


def make_chat(settings, indexed, client):
    store, vectors, nlp = indexed
    return Chat(settings, Retriever(settings, store, vectors, nlp), client)


def test_query_followups_and_literal_failure(settings, indexed):
    client = Client(PLAN)
    chat = make_chat(settings, indexed, client)
    history = [{"role": "user", "content": "Tell me about coastal flooding"}]
    prepared = chat.prepare("What happened next?", history)
    assert prepared.queries == ["coastal flood"]
    assert history[0] in client.calls[0]["messages"]
    client = Client(ConnectionError("offline"))
    fallback = make_chat(settings, indexed, client).prepare(
        "flood", history, allow_literal=True
    )
    assert fallback.queries == ["flood"]
    assert "literal hybrid" in fallback.warning
    with pytest.raises(ChatError, match="interpret"):
        make_chat(settings, indexed, Client(ConnectionError("offline"))).prepare("flood", history)


def test_grounded_answer_and_citation_retry(settings, indexed):
    client = Client(
        ["Floods affected coastal areas [99]."],
        ["Flood protection was discussed ", "[1]."],
    )
    reply = make_chat(settings, indexed, client).answer("What happened during the flood?")
    assert reply.text.endswith("[1].")
    assert reply.sources
    assert len(client.calls) == 2
    assert all(call["stream"] is True and "format" not in call for call in client.calls)
    assert all(call["keep_alive"] == "30m" for call in client.calls)


def test_invalid_citations_are_explicit_error(settings, indexed):
    client = Client(
        ["Missing citations."],
        ["Invented citations [100]."],
    )
    with pytest.raises(ChatError, match="valid source citations"):
        make_chat(settings, indexed, client).answer("What happened?")


def test_empty_candidates_do_not_generate(settings, indexed, monkeypatch):
    store, vectors, _ = indexed
    monkeypatch.setattr(store, "lexical", lambda *args: [])
    monkeypatch.setattr(vectors, "search", lambda *args: [])
    monkeypatch.setattr(
        Chat,
        "prepare",
        lambda *args, **kwargs: PreparedQuery("Fixture question", ["fixture query"]),
    )
    client = Client()
    reply = make_chat(settings, indexed, client).answer("Fixture question")
    assert reply.text == NO_EVIDENCE
    assert reply.sources == []
    assert not client.calls


@pytest.mark.parametrize("supported", [True, False])
def test_low_scoring_evidence_reaches_generation(settings, indexed, monkeypatch, supported):
    _, _, nlp = indexed
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [-100.0] * len(pairs))
    monkeypatch.setattr(
        Chat,
        "prepare",
        lambda *args, **kwargs: PreparedQuery("Fixture question", ["fixture query"]),
    )
    client = Client(["Fixture answer [1]."] if supported else [NO_EVIDENCE])
    reply = make_chat(settings, indexed, client).answer("Fixture question")
    assert len(client.calls) == 1
    assert "NEWS EVIDENCE:" in client.calls[0]["messages"][-1]["content"]
    if supported:
        assert reply.text == "Fixture answer [1]."
        assert reply.sources
        assert all(hit.score == -100.0 for hit in reply.sources)
    else:
        assert reply.text == NO_EVIDENCE
        assert reply.sources == []


def test_article_scope_supplies_full_body(settings, indexed):
    store, vectors, nlp = indexed
    hit = Retriever(settings, store, vectors, nlp).search(["flood"])[0]
    article = store.article(hit.article.id)
    client = Client(["It discusses flood protection [1]."])
    chat = make_chat(settings, indexed, client)
    reply = chat.answer("Summarize this article", filters=Filters(article_id=article.id))
    assert len(reply.sources) == 1
    assert reply.sources[0].article.id == article.id
    assert article.body in client.calls[0]["messages"][-1]["content"]


def test_model_refusal_does_not_display_unsupported_text(settings, indexed):
    client = Client([NO_EVIDENCE])
    reply = make_chat(settings, indexed, client).answer("What happened?")
    assert reply.text == NO_EVIDENCE
    assert not reply.sources


def test_malformed_plan_and_bounded_history(settings, indexed):
    client = Client({"not_a_plan": True})
    with pytest.raises(ChatError, match="interpret"):
        make_chat(settings, indexed, client).prepare(
            "Flood?", [{"role": "user", "content": "Earlier question"}]
        )
    settings = replace(settings, context_tokens=4096)
    client = Client(PLAN)
    history = [{"role": "user", "content": "old question " * 1000}] * 20
    make_chat(settings, indexed, client).prepare("Flood?", history)
    assert len(client.calls[0]["messages"]) < 8


def test_oversized_multibyte_question_is_reported(settings, indexed):
    chat = make_chat(replace(settings, context_tokens=4096), indexed, Client(PLAN))
    with pytest.raises(ChatError):
        chat.prepare("\u6d2a" * 1500, [{"role": "user", "content": "Earlier question"}])


def test_query_rewrite_retains_explicit_constraints(settings, indexed):
    client = Client(PLAN)
    query = make_chat(settings, indexed, client).prepare(
        'flood 2026 "New York"', [{"role": "user", "content": "Earlier question"}]
    )
    assert "2026" in query.queries[0]
    assert '"New York"' in query.queries[0]


def test_long_article_chat_discloses_partial_evidence(settings, nlp):
    settings = replace(settings, context_tokens=8192)
    body = "Flood protection for coastal communities. " * 2000
    ingest(settings, rows=[source_row(0, body=body)], nlp=nlp)
    retriever = Retriever(settings, nlp=nlp)
    hit = retriever.search(["coastal flood"])[0]
    client = Client(["It covers flood protection [1]."])
    chat = Chat(settings, retriever, client)
    reply = chat.answer("What happened?", filters=Filters(article_id=hit.article.id))
    assert "retrieved excerpts" in reply.notice
    assert body not in client.calls[-1]["messages"][-1]["content"]
    assert reply.sources[0].article.id == hit.article.id


def test_first_turn_skips_rewrite_and_streams_progressively(settings, indexed):
    client = Client(["Coastal ", "flooding ", "was reported [1]."])
    chat = make_chat(settings, indexed, client)
    assert chat.prepare('flood 2021 "coastal flooding"').queries == [
        'flood 2021 "coastal flooding"'
    ]
    assert not client.calls
    updates = []
    reply = chat.answer("What happened during coastal flooding?", on_text=updates.append)
    assert updates == [
        "",
        "Coastal ",
        "Coastal flooding ",
        "Coastal flooding was reported [1].",
    ]
    assert reply.text == updates[-1]
    assert len(client.calls) == 1
    assert client.calls[0]["options"]["num_ctx"] == 64000


def test_followup_still_rewrites_and_keeps_model_alive(settings, indexed):
    client = Client(PLAN, ["Flood protection was reported [1]."])
    history = [{"role": "user", "content": "Tell me about coastal flooding."}]
    make_chat(settings, indexed, client).answer("What happened next?", history)
    assert len(client.calls) == 2
    assert client.calls[0]["stream"] is False
    assert "format" in client.calls[0]
    assert all(call["keep_alive"] == "30m" for call in client.calls)


def test_failed_stream_clears_draft_and_reports_error(settings, indexed):
    client = Client(["Incomplete answer ", ConnectionError("stream interrupted")])
    updates = []
    with pytest.raises(ChatError, match="stream interrupted"):
        make_chat(settings, indexed, client).answer("flood", on_text=updates.append)
    assert updates == ["", "Incomplete answer ", ""]


def test_rejected_citation_drafts_are_cleared(settings, indexed):
    client = Client(["Invalid [999]."], ["Still invalid [999]."])
    updates = []
    with pytest.raises(ChatError, match="valid source citations"):
        make_chat(settings, indexed, client).answer("flood", on_text=updates.append)
    assert updates[-1] == ""
    assert updates.count("Invalid [999].") == 1


def test_empty_stream_is_an_explicit_error(settings, indexed):
    with pytest.raises(ChatError, match="empty answer"):
        make_chat(settings, indexed, Client([])).answer("flood")


def test_warmup_loads_model_with_context_and_keep_alive(settings, indexed):
    client = Client({})
    make_chat(settings, indexed, client).warmup()
    assert len(client.calls) == 1
    assert client.calls[0]["messages"] == []
    assert client.calls[0]["keep_alive"] == "30m"
    assert client.calls[0]["options"]["num_ctx"] == 64000


@pytest.mark.parametrize("reason", [None, "length"])
def test_incomplete_or_truncated_streams_cannot_be_accepted(settings, indexed, reason):
    class TruncatedClient:
        def chat(self, **kwargs):
            yield SimpleNamespace(
                message=SimpleNamespace(content="Partial but cited [1]."), done=False
            )
            if reason:
                yield SimpleNamespace(
                    message=SimpleNamespace(content=""), done=True, done_reason=reason
                )

    updates = []
    with pytest.raises(ChatError, match="length limit|before completing"):
        make_chat(settings, indexed, TruncatedClient()).answer("flood", on_text=updates.append)
    assert updates[-1] == ""

from pathlib import Path
from threading import Event

import pytest
from streamlit.testing.v1 import AppTest

from news_recommender.chat import Chat, Reply
from news_recommender.ingestion import ingest
from news_recommender.nlp import NLP
from news_recommender.retrieval import Retriever
from news_recommender.services import Services
from news_recommender.storage import Filters, Store, Vectors

from .conftest import source_row


@pytest.fixture(autouse=True)
def no_model_warmup(monkeypatch, nlp):
    monkeypatch.setattr(NLP, "warmup", lambda self: None)
    monkeypatch.setattr(NLP, "embed_queries", lambda self, queries: nlp.embed_queries(queries))
    monkeypatch.setattr(NLP, "ranker", nlp.ranker)
    monkeypatch.setattr(NLP, "rerank", lambda self, pairs: nlp.rerank(pairs))
    monkeypatch.setattr(Chat, "warmup", lambda self: None)
    original_init = Services.__init__

    def ready_services(self, settings):
        original_init(self, settings)
        self.retrieval.result(timeout=5)
        self.chat.result(timeout=5)

    monkeypatch.setattr(Services, "__init__", ready_services)
    return original_init


def test_search_reader_and_scoped_chat(settings, indexed, monkeypatch):
    store, vectors, nlp = indexed
    hits = Retriever(settings, store, vectors, nlp).search(["flood"])
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    monkeypatch.setattr(Retriever, "search", lambda *args, **kwargs: hits)
    monkeypatch.setattr(
        Chat,
        "answer",
        lambda *args, **kwargs: Reply("The article discusses flood protection [1].", hits[:1]),
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    assert not app.sidebar.subheader
    assert not any(
        item.value in ("Search ready", "Chat ready") for item in app.caption
    )
    assert "#### What would you like to read about?" in [
        element.value for element in app.markdown
    ]
    assert app.text_input[0].label == "What would you like to read about?"
    assert app.text_input[0].proto.label_visibility.value == 2
    assert not any(
        "Hybrid full-text search, complete articles, and source-grounded conversation."
        in element.value
        for element in app.caption
    )
    app.text_input[0].set_value("coastal flood")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    assert len([button for button in app.button if button.label == "Read full article"]) == 5
    next(button for button in app.button if button.label == "Read full article").click().run()
    assert store.article(hits[0].article.id).body in [element.value for element in app.text]
    next(button for button in app.button if button.label == "Ask about this article").click().run()
    assert app.radio[0].value == "Chat"
    app.chat_input[0].set_value("What is this about?").run()
    assert not app.exception
    assert any("flood protection [1]" in element.value for element in app.markdown)
    next(button for button in app.button if button.label == "Clear conversation").click().run()
    assert not app.session_state["messages"]


def test_search_failure_is_not_reported_as_zero_matches(settings, indexed, monkeypatch):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))

    def fail(*args, **kwargs):
        raise RuntimeError("Embedding model failed")

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    app.session_state["searched"] = True
    app.run()
    monkeypatch.setattr(Retriever, "search", fail)
    app.text_input[0].set_value("flood")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert app.error and "Embedding model failed" in app.error[0].value
    assert not app.session_state["searched"]
    assert not any("Found" in item.value for item in app.info)


@pytest.mark.parametrize("candidate_count", [0, 2, 7])
def test_search_displays_ranked_candidates(settings, indexed, monkeypatch, candidate_count):
    store, vectors, nlp = indexed
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    monkeypatch.setattr(nlp, "rerank", lambda pairs: [-100.0] * len(pairs))
    monkeypatch.setattr(store, "lexical", lambda *args: [])
    with store.connect() as db:
        ids = [
            row[0]
            for row in db.execute("SELECT id FROM chunks ORDER BY id LIMIT ?", (candidate_count,))
        ]
    monkeypatch.setattr(vectors, "search", lambda *args: ids)
    original_init = Retriever.__init__

    def initialize(self, settings, **kwargs):
        original_init(self, settings, store, vectors, nlp)

    monkeypatch.setattr(Retriever, "__init__", initialize)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("Search must not invoke the conversational query interpreter.")

    monkeypatch.setattr(Chat, "prepare", unexpected_prepare)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.text_input[0].set_value("Fixture question")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    expected_count = candidate_count
    assert (
        len([button for button in app.button if button.label == "Read full article"])
        == expected_count
    )
    notices = [item.value for item in [*app.info, *app.success]]
    expected_notice = f"Found {expected_count} articles."
    assert any(expected_notice in value for value in notices)
    assert not any("qualifying" in value or "weak matches" in value for value in notices)
    if expected_count:
        assert all(hit.score == -100.0 for hit in app.session_state["results"])
    assert not any(
        phrase in item.value
        for item in app.caption
        for phrase in ("Showing at most the top", "Phrase matches first", "not guaranteed relevant")
    )


def test_chat_caption_matches_requested_wording(settings, indexed, monkeypatch):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.radio[0].set_value("Chat").run()
    assert not app.exception
    assert "Answers are based on the current dataset version and the sidebar filters." in [
        item.value for item in app.caption
    ]


def test_search_preserves_original_query_and_requests_lexical_priority(
    settings, indexed, monkeypatch
):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    calls = []

    def search(self, queries, filters, *, limit=5, prefer_lexical=False):
        calls.append((queries, filters, prefer_lexical, limit))
        return []

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("Search must not rewrite user input.")

    monkeypatch.setattr(Chat, "prepare", unexpected_prepare)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    monkeypatch.setattr(Retriever, "search", search)
    query = 'FixtureAlpha "FixtureBeta FixtureGamma"'
    app.text_input[0].set_value(f"  {query}  ")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert calls[0][0] == [query]
    assert calls[0][2] is True
    assert calls[0][3] == settings.search_limit


def test_search_rejects_blank_input_without_retrieval(settings, indexed, monkeypatch):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))

    def unexpected_search(*args, **kwargs):
        pytest.fail("Blank input must be rejected before retrieval.")

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    monkeypatch.setattr(Retriever, "search", unexpected_search)
    app.text_input[0].set_value("   ")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    assert app.error and "between 1 and 1500 characters" in app.error[0].value
    assert not app.session_state["searched"]


def test_search_numbered_pages_reader_and_resets(settings, nlp, monkeypatch):
    vectors = Vectors(settings, create=True)
    ingest(settings, rows=[source_row(index) for index in range(23)], nlp=nlp, vectors=vectors)
    hits = Retriever(settings, Store(settings.database), vectors, nlp).search(
        ["flood"], limit=None, prefer_lexical=True
    )
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    calls = []

    def search(*args, **kwargs):
        calls.append(kwargs)
        return hits

    monkeypatch.setattr(Retriever, "search", search)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    calls.clear()

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    def headlines():
        return [element.value for element in app.main.subheader]

    app.text_input[0].set_value("flood")
    click("Find articles")
    assert "Found 23 articles." in app.success[0].value
    assert headlines() == [hit.article.headline for hit in hits[:10]]
    assert next(button for button in app.button if button.label == "Previous").disabled
    click("2")
    assert headlines() == [hit.article.headline for hit in hits[10:20]]
    click("Read full article")
    body = Store(settings.database).article(hits[10].article.id).body
    assert body in [element.value for element in app.text]
    click("Back to results")
    assert headlines() == [hit.article.headline for hit in hits[10:20]]
    click("Next")
    assert headlines() == [hit.article.headline for hit in hits[20:]]
    assert next(button for button in app.button if button.label == "Next").disabled
    assert any("Page 3 of 3" in item.value for item in app.caption)
    click("Previous")
    assert headlines() == [hit.article.headline for hit in hits[10:20]]
    assert len(calls) == 1
    click("Find articles")
    assert app.session_state["search_page"] == 1
    assert headlines() == [hit.article.headline for hit in hits[:10]]
    click("3")
    app.selectbox[0].select("news").run()
    assert not app.exception
    assert not app.session_state["results"]
    assert not app.session_state["searched"]
    assert app.session_state["search_page"] == 1
    assert not headlines()


def test_numbered_pagination_can_jump_to_distant_pages(settings, indexed, monkeypatch):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from app import search_pagination\n"
        "search_pagination(st.session_state.get('search_page', 1), 100)\n"
    ).run()
    assert not app.exception
    assert len(app.button) < 10
    assert not app.get("column")
    layout = str(app.get("flex_container")[-1].proto.flex_container)
    assert "direction: HORIZONTAL" in layout
    assert "gap_size: SMALL" in layout
    assert "justify: JUSTIFY_START" in layout
    assert any(element.value == "..." for element in app.markdown)
    next(button for button in app.button if button.label == "100").click().run()
    assert not app.exception
    assert app.session_state["search_page"] == 100
    assert next(button for button in app.button if button.label == "Next").disabled
    next(button for button in app.button if button.label == "1").click().run()
    assert not app.exception
    assert app.session_state["search_page"] == 1


def test_services_warm_models_and_search_once_in_cached_startup(settings, indexed, monkeypatch):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    calls = []
    searches = []
    retrievers = []
    original_search = Retriever.search

    def search(self, queries, filters=None, *, limit=5, prefer_lexical=False):
        calls.append("search")
        searches.append((queries, filters, limit, prefer_lexical))
        retrievers.append(self)
        hits = original_search(
            self, queries, filters, limit=limit, prefer_lexical=prefer_lexical
        )
        assert hits
        return hits

    monkeypatch.setattr(NLP, "warmup", lambda self: calls.append("nlp"))
    monkeypatch.setattr(Retriever, "search", search)
    monkeypatch.setattr(Chat, "warmup", lambda self: calls.append("ollama"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    app.run()
    assert not app.exception
    assert calls == ["nlp", "search", "ollama"]
    assert searches == [(["news"], Filters(), settings.search_limit, True)]
    assert retrievers[0].vectors._search_count == indexed[1].count()
    assert not app.session_state["results"]
    assert not app.session_state["searched"]


def test_failed_stream_draft_is_not_persisted(settings, indexed, monkeypatch):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))

    def fail(self, *args, on_text, **kwargs):
        on_text("Invalid draft [999].")
        raise RuntimeError("Stream failed")

    monkeypatch.setattr(Chat, "answer", fail)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.radio[0].set_value("Chat").run()
    app.chat_input[0].set_value("flood").run()
    assert not app.exception
    assert app.error and "Stream failed" in app.error[0].value
    assert not app.session_state["messages"]
    assert all("Invalid draft" not in element.value for element in app.markdown)


def test_ollama_warmup_failure_keeps_search_available(settings, indexed, monkeypatch):
    from news_recommender.chat import ChatError

    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))

    def fail(self):
        raise ChatError("Ollama is offline. Search remains available.")

    monkeypatch.setattr(Chat, "warmup", fail)
    monkeypatch.setattr(Retriever, "search", lambda *args, **kwargs: [])
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    assert not app.exception
    app.text_input[0].set_value("flood")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    app.radio[0].set_value("Chat").run()
    assert any("Ollama is offline" in item.value for item in app.warning)


def test_search_works_while_ollama_warms_in_background(
    settings, indexed, monkeypatch, no_model_warmup
):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    started, release = Event(), Event()
    instances = []
    calls = []

    def initialize(self, settings):
        no_model_warmup(self, settings)
        instances.append(self)

    def warmup(self):
        calls.append("ollama")
        started.set()
        assert release.wait(timeout=10)

    monkeypatch.setattr(Services, "__init__", initialize)
    monkeypatch.setattr(Chat, "warmup", warmup)
    monkeypatch.setattr(Retriever, "search", lambda *args, **kwargs: [])
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        assert not app.exception
        assert app.title[0].value == "News Explorer"
        assert app.text_input
        assert started.wait(timeout=5)
        assert not release.is_set()
        app.run()
        assert not next(button for button in app.button if button.label == "Find articles").disabled
        assert not any("Chat: Loading Ollama model" in item.value for item in app.info)
        app.text_input[0].set_value("flood")
        next(button for button in app.button if button.label == "Find articles").click().run()
        assert not app.exception
        assert any("Found 0 articles" in item.value for item in app.info)
        app.radio[0].set_value("Chat").run()
        assert app.chat_input[0].disabled
        assert any("Chat: Loading Ollama model" in item.value for item in app.info)
    finally:
        release.set()
        for instance in instances:
            instance.chat.result(timeout=5)
    app.run()
    assert not app.chat_input[0].disabled
    assert not any(item.value == "Chat ready" for item in app.caption)
    assert not any("Chat: Loading Ollama model" in item.value for item in app.info)
    assert calls == ["ollama"]


@pytest.mark.parametrize(
    ("target", "method", "message"),
    [
        (NLP, "warmup", "Loading embedding model"),
        (Retriever, "search", "Warming keyword and vector search"),
    ],
)
def test_page_and_article_reader_render_while_retrieval_loads(
    settings, indexed, monkeypatch, no_model_warmup, target, method, message
):
    store, _, _ = indexed
    article_id = store.lexical('"flood"', Filters(), 1)[0].article_id
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    started, release = Event(), Event()
    instances = []
    original = getattr(target, method)

    def initialize(self, settings):
        no_model_warmup(self, settings)
        instances.append(self)

    def warmup(self, *args, **kwargs):
        started.set()
        assert release.wait(timeout=10)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Services, "__init__", initialize)
    monkeypatch.setattr(target, method, warmup)
    try:
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
        app.session_state["selected_article"] = article_id
        app.run()
        assert not app.exception
        assert started.wait(timeout=5)
        assert not release.is_set()
        app.run()
        assert not app.exception
        assert next(button for button in app.button if button.label == "Find articles").disabled
        assert any(f"Search: {message}" in item.value for item in app.info)
        assert store.article(article_id).body in [item.value for item in app.text]
        app.radio[0].set_value("Chat").run()
        assert app.chat_input[0].disabled
    finally:
        release.set()
        for instance in instances:
            instance.chat.result(timeout=5)
    app.radio[0].set_value("Search").run()
    assert not app.exception
    assert not next(button for button in app.button if button.label == "Find articles").disabled
    assert not app.session_state["results"]
    assert not app.session_state["searched"]


@pytest.mark.parametrize(
    ("target", "method", "message"),
    [
        (NLP, "warmup", "Embedding startup failed"),
        (Retriever, "search", "Search warmup failed"),
        (Vectors, "search", "Vector startup failed"),
    ],
)
def test_retrieval_startup_failure_is_visible_without_blocking_the_page(
    settings, indexed, monkeypatch, no_model_warmup, target, method, message
):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    instances = []

    def initialize(self, settings):
        no_model_warmup(self, settings)
        instances.append(self)

    def fail(self, *args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(Services, "__init__", initialize)
    monkeypatch.setattr(target, method, fail)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    for future in (instances[0].retrieval, instances[0].chat):
        with pytest.raises(RuntimeError, match=message):
            future.result(timeout=5)
    app.run()
    assert not app.exception
    assert app.title[0].value == "News Explorer"
    assert any(message in item.value for item in app.error)
    assert next(button for button in app.button if button.label == "Find articles").disabled

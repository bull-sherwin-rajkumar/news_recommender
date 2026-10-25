from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from news_recommender.chat import Chat, Reply
from news_recommender.retrieval import Retriever


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
    app.text_input[0].set_value("coastal flood")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    assert len([button for button in app.button if button.label == "Read full article"]) == 5
    next(button for button in app.button if button.label == "Read full article").click().run()
    assert hits[0].article.body in [element.value for element in app.text]
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

    monkeypatch.setattr(Retriever, "search", fail)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"))
    app.session_state["searched"] = True
    app.run()
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

    def initialize(self, settings):
        original_init(self, settings, store, vectors, nlp)

    monkeypatch.setattr(Retriever, "__init__", initialize)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("Search must not invoke the conversational query interpreter.")

    monkeypatch.setattr(Chat, "prepare", unexpected_prepare)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.text_input[0].set_value("Fixture question")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    expected_count = min(candidate_count, 5)
    assert (
        len([button for button in app.button if button.label == "Read full article"])
        == expected_count
    )
    notices = [item.value for item in [*app.info, *app.success]]
    expected_notice = (
        f"Found {expected_count} article(s)." if expected_count < 5 else "Found 5 articles."
    )
    assert any(expected_notice in value for value in notices)
    assert not any("qualifying" in value or "weak matches" in value for value in notices)
    if expected_count:
        assert all(hit.score == -100.0 for hit in app.session_state["results"])
        assert any("not guaranteed relevant" in item.value for item in app.caption)


def test_search_preserves_original_query_and_requests_lexical_priority(
    settings, indexed, monkeypatch
):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))
    calls = []

    def search(self, queries, filters, *, prefer_lexical=False):
        calls.append((queries, filters, prefer_lexical))
        return []

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("Search must not rewrite user input.")

    monkeypatch.setattr(Retriever, "search", search)
    monkeypatch.setattr(Chat, "prepare", unexpected_prepare)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    query = 'FixtureAlpha "FixtureBeta FixtureGamma"'
    app.text_input[0].set_value(f"  {query}  ")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert calls[0][0] == [query]
    assert calls[0][2] is True


def test_search_rejects_blank_input_without_retrieval(settings, indexed, monkeypatch):
    monkeypatch.setenv("NEWS_DATA_DIR", str(settings.data_dir))

    def unexpected_search(*args, **kwargs):
        pytest.fail("Blank input must be rejected before retrieval.")

    monkeypatch.setattr(Retriever, "search", unexpected_search)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
    app.text_input[0].set_value("   ")
    next(button for button in app.button if button.label == "Find articles").click().run()
    assert not app.exception
    assert app.error and "between 1 and 1500 characters" in app.error[0].value
    assert not app.session_state["searched"]

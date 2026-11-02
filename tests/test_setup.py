from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dotenv import dotenv_values

from news_recommender.chat import Chat, ChatError
from news_recommender.config import Settings
from news_recommender.services import Services


def run_startup():
    script = (Path(__file__).parents[1] / "setup.sh").read_text()
    marker = 'echo "Checking retrieval and Ollama initialization..."\n.venv/bin/python <<\'PY\'\n'
    source = script.split(marker, 1)[1].split("\nPY\n", 1)[0]
    exec(compile(source, "setup.sh:startup", "exec"), {})


@pytest.fixture
def startup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = tmp_path / ".env"
    env.write_text(
        "NEWS_CONTEXT_TOKENS=64000\n"
        "NEWS_EMBEDDING_MODEL=custom/embedding\n"
        "NEWS_RERANKER_MODEL=custom/reranker\n"
    )
    monkeypatch.setattr(Settings, "from_env", lambda: Settings())
    retrieval = Future()
    retrieval.set_result(object())
    chat = Future()
    chat.set_result(SimpleNamespace(warmup_warning=None))
    monkeypatch.setattr(
        Services, "__init__", lambda self, settings: self.__dict__.update(
            retrieval=retrieval, chat=chat
        )
    )
    attempts = []

    def warmup(self):
        attempts.append(self.settings.context_tokens)

    monkeypatch.setattr(Chat, "warmup", warmup)
    return SimpleNamespace(
        env=env, chat=chat.result(), attempts=attempts, retrieval=retrieval
    )


def test_setup_keeps_successful_context(startup):
    before = startup.env.read_text()
    run_startup()
    assert startup.attempts == []
    assert startup.env.read_text() == before


def test_setup_persists_successful_8k_fallback(startup, capsys):
    startup.chat.warmup_warning = "Ollama ran out of memory"
    run_startup()
    assert startup.attempts == [8192]
    assert dotenv_values(startup.env) == {
        "NEWS_CONTEXT_TOKENS": "8192",
        "NEWS_EMBEDDING_MODEL": "custom/embedding",
        "NEWS_RERANKER_MODEL": "custom/reranker",
    }
    assert "Retrying Ollama" in capsys.readouterr().err


def test_setup_failed_fallback_does_not_change_env(startup, monkeypatch):
    before = startup.env.read_text()
    startup.chat.warmup_warning = "Ollama ran out of memory"
    warmup = Mock(side_effect=ChatError("8k failed"))
    monkeypatch.setattr(Chat, "warmup", warmup)
    with pytest.raises(ChatError, match="8k failed"):
        run_startup()
    warmup.assert_called_once()
    assert startup.env.read_text() == before


def test_setup_preserves_custom_context(startup, monkeypatch):
    monkeypatch.setattr(Settings, "from_env", lambda: Settings(context_tokens=16384))
    startup.chat.warmup_warning = "Ollama unavailable"
    before = startup.env.read_text()
    with pytest.raises(RuntimeError, match="Ollama unavailable"):
        run_startup()
    assert startup.attempts == []
    assert startup.env.read_text() == before


def test_setup_does_not_retry_retrieval_failure(startup, monkeypatch):
    monkeypatch.setattr(
        startup.retrieval, "result", Mock(side_effect=RuntimeError("retrieval failed"))
    )
    with pytest.raises(RuntimeError, match="retrieval failed"):
        run_startup()
    assert startup.attempts == []

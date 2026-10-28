from types import SimpleNamespace

import numpy as np
import pytest

from news_recommender.nlp import NLP, embedding_model, reranking_model


def test_cuda_requirement_is_explicit(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    embedding_model.cache_clear()
    with pytest.raises(RuntimeError, match="NEWS_DEVICE=cpu"):
        embedding_model("fixture-model", "cuda")


def test_onnx_reranker_runs_on_cpu_and_keeps_truncation(monkeypatch):
    calls = []

    def cross_encoder(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr("sentence_transformers.CrossEncoder", cross_encoder)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *args, **kwargs: "/cached/model"
    )
    reranking_model.cache_clear()
    reranking_model("fixture-model", "cuda", "onnx")
    assert calls == [
        (
            "/cached/model",
            {
                "device": "cpu",
                "backend": "onnx",
                "model_kwargs": {
                    "provider": "CPUExecutionProvider",
                    "file_name": "onnx/model.onnx",
                },
                "max_length": 512,
            },
        )
    ]
    reranking_model.cache_clear()


def test_rerank_does_not_tokenize_before_predict(settings):
    calls = []

    def predict(pairs, **kwargs):
        calls.append((pairs, kwargs))
        return np.array([0.75])

    nlp = NLP(settings)
    nlp.ranker = SimpleNamespace(predict=predict)
    pairs = [("query", "passage " * 1000)]
    assert nlp.rerank(pairs) == [0.75]
    assert calls == [(pairs, {"batch_size": settings.batch_size, "show_progress_bar": False})]


def test_warmup_runs_embedding_and_reranker(settings, monkeypatch):
    nlp = NLP(settings)
    calls = []
    monkeypatch.setattr(nlp, "embed_queries", lambda queries: calls.append(("embed", queries)))
    monkeypatch.setattr(nlp, "rerank", lambda pairs: calls.append(("rerank", pairs)))
    nlp.warmup()
    assert [name for name, _ in calls] == ["embed", "rerank"]

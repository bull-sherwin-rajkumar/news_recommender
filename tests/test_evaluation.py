import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from news_recommender import evaluation
from news_recommender.chat import Reply
from news_recommender.evaluation import Case, metrics


def test_precision_and_ranking_metrics():
    relevant = {str(index): 2 for index in range(5)}
    assert metrics(list(relevant), relevant) == {"precision_at_5": 1.0, "ndcg_at_5": 1.0}
    assert metrics(["unknown"], relevant) == {"precision_at_5": 0.0, "ndcg_at_5": 0.0}
    assert metrics(["0"], relevant)["precision_at_5"] == 0.2


def test_label_validation():
    with pytest.raises(ValidationError):
        Case(query="flood", relevant={"id": 3})
    with pytest.raises(ValidationError):
        Case(query="flood", relevant={"id": 1}, unsupported=True)
    with pytest.raises(ValidationError):
        Case(query="flood")


def test_saved_labels_have_separate_valid_splits():
    path = Path(__file__).with_name("relevance_labels.json")
    cases = [Case.model_validate(item) for item in json.loads(path.read_text())]
    assert len(cases) == 13
    assert all(not case.unsupported for case in cases)
    calibration = {case.query for case in cases if case.split == "calibration"}
    evaluation = {case.query for case in cases if case.split == "evaluation"}
    assert len(calibration) == 3 and len(evaluation) == 10
    assert calibration.isdisjoint(evaluation)
    assert all(len(article_id) == 32 for case in cases for article_id in case.relevant)
    assert sum(case.query_type == "keyword" for case in cases) == 12
    assert sum(case.query_type == "natural_language" for case in cases) == 1


@pytest.mark.parametrize("include_unsupported", [False, True])
def test_report_preserves_scores_without_threshold(
    tmp_path, monkeypatch, capsys, include_unsupported
):
    labels = tmp_path / "labels.json"
    relevant = {f"candidate-{index}": 2 for index in range(5)}
    cases = [{"query": "fixture query", "relevant": relevant}]
    if include_unsupported:
        cases.append(
            {
                "query": "fixture query",
                "unsupported": True,
                "filters": {"section": "empty"},
            }
        )
    labels.write_text(json.dumps(cases))
    hits = [
        SimpleNamespace(
            article=SimpleNamespace(id=article_id, headline=f"Fixture {index}"),
            score=float(-10 - index),
        )
        for index, article_id in enumerate(relevant)
    ]

    def search(queries, filters, *, method):
        assert queries == ["fixture query"]
        assert method == "hybrid"
        return [] if filters.section == "empty" else hits

    monkeypatch.setattr(evaluation, "Retriever", lambda settings: SimpleNamespace(search=search))
    monkeypatch.setattr("sys.argv", ["news-evaluate", "--labels", str(labels)])
    evaluation.main()
    report = json.loads(capsys.readouterr().out)
    assert "threshold" not in report
    assert report["method"] == "hybrid"
    assert report["split"] == "evaluation"
    assert report["mean_precision_at_5"] == report["mean_ndcg_at_5"] == 1.0
    assert "unsupported_accuracy" not in report
    assert all("correct_abstention" not in result for result in report["results"])
    assert [hit["score"] for hit in report["results"][0]["hits"]] == [hit.score for hit in hits]
    assert len(report["results"]) == (2 if include_unsupported else 1)


def test_all_stages_report_chat_review_followups_and_latencies(tmp_path, monkeypatch, capsys):
    labels = tmp_path / "labels.json"
    relevant = {f"candidate-{index}": 2 for index in range(5)}
    labels.write_text(
        json.dumps(
            [
                {
                    "query": "fixture query",
                    "query_type": "keyword",
                    "relevant": relevant,
                    "expected_facts": ["The event happened in 2024."],
                    "follow_ups": [
                        {
                            "query": "What happened next?",
                            "expected_facts": ["The response began the following day."],
                        }
                    ],
                }
            ]
        )
    )
    hits = [
        SimpleNamespace(
            article=SimpleNamespace(id=article_id, headline=f"Fixture {index}"),
            score=1.0,
        )
        for index, article_id in enumerate(relevant)
    ]
    search_calls = []

    def search(queries, filters, *, method):
        search_calls.append(queries)
        return hits

    retriever = SimpleNamespace(
        nlp=SimpleNamespace(warmup=lambda: None),
        search=search,
    )
    answer_calls = []

    class FakeChat:
        def __init__(self, settings, actual_retriever):
            assert actual_retriever is retriever

        warmup_warning = None

        def warmup(self):
            pass

        def answer(self, query, history, filters):
            answer_calls.append((query, list(history)))
            return Reply("Fixture answer [1].", hits[:1], generation_seconds=0.25)

    monkeypatch.setattr(evaluation.Settings, "from_env", lambda: object())
    monkeypatch.setattr(evaluation, "Retriever", lambda settings: retriever)
    monkeypatch.setattr(evaluation, "Chat", FakeChat)
    chat = FakeChat(None, retriever)
    monkeypatch.setattr(
        evaluation,
        "Services",
        lambda settings: SimpleNamespace(
            retrieval=SimpleNamespace(result=lambda: retriever),
            chat=SimpleNamespace(result=lambda: chat),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["news-evaluate", "--labels", str(labels), "--stage", "all"],
    )
    evaluation.main()

    report = json.loads(capsys.readouterr().out)
    assert report["stages"]["test_set_preparation"]["query_mix"] == {"keyword": 1}
    assert report["stages"]["retrieval"]["mean_precision_at_5"] == 1.0
    assert report["stages"]["chat_answers"]["human_review_required"] is True
    assert report["results"][0]["chat"]["turns"][0]["expected_facts"] == [
        "The event happened in 2024."
    ]
    assert report["results"][0]["chat"]["turns"][0]["manual_review"]["correctness"] is None
    assert report["stages"]["latencies"]["mean_llm_generation_seconds"] == 0.25
    assert search_calls == [["fixture query"]]
    assert answer_calls[1][0] == "What happened next?"
    assert [message["role"] for message in answer_calls[1][1]] == ["user", "assistant"]


def test_removed_threshold_option_is_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["news-evaluate", "--labels", str(tmp_path / "labels.json"), "--threshold", "0"],
    )
    with pytest.raises(SystemExit) as exc:
        evaluation.main()
    assert exc.value.code == 2
    assert "unrecognized arguments: --threshold" in capsys.readouterr().err

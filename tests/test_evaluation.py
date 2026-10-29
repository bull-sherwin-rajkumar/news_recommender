import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from news_recommender import evaluation
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


def test_removed_threshold_option_is_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["news-evaluate", "--labels", str(tmp_path / "labels.json"), "--threshold", "0"],
    )
    with pytest.raises(SystemExit) as exc:
        evaluation.main()
    assert exc.value.code == 2
    assert "unrecognized arguments: --threshold" in capsys.readouterr().err

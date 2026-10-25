import argparse
import json
import math
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .config import Settings
from .retrieval import Retriever
from .storage import Filters


class Case(BaseModel):
    query: str = Field(min_length=1)
    split: str = "evaluation"
    relevant: dict[str, int] = Field(default_factory=dict)
    unsupported: bool = False
    filters: dict[str, str | int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_grades(self) -> "Case":
        if any(grade not in (0, 1, 2) for grade in self.relevant.values()):
            raise ValueError("Relevance grades must be 0 (irrelevant), 1, or 2.")
        if self.unsupported and any(self.relevant.values()):
            raise ValueError("Unsupported cases cannot contain positive relevance labels.")
        if not self.unsupported and not any(self.relevant.values()):
            raise ValueError("Supported cases require positive relevance labels.")
        return self


def metrics(ids: list[str], relevant: dict[str, int]) -> dict[str, float]:
    grades = [relevant.get(article_id, 0) for article_id in ids[:5]]
    ideal = sorted(relevant.values(), reverse=True)[:5]

    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))

    return {
        "precision_at_5": sum(grade > 0 for grade in grades) / 5,
        "ndcg_at_5": dcg(grades) / dcg(ideal) if dcg(ideal) else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate literal retrieval independently of LLM query rewriting. Labels JSON is "
            "a list of {query, split, relevant: {article_id: 0|1|2}, unsupported, filters}. "
            "Use --split to select a saved label split. Unsupported-query abstention "
            "measures empty retrieval, not Chat's evidence-based refusal."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--method", choices=("hybrid", "bm25"), default="hybrid")
    args = parser.parse_args()
    settings = Settings.from_env()
    cases = [Case.model_validate(item) for item in json.loads(args.labels.read_text())]
    cases = [case for case in cases if case.split == args.split]
    if not cases:
        parser.error(f"No cases for split {args.split!r}.")
    retriever = Retriever(settings)
    results = []
    for case in cases:
        hits = retriever.search([case.query], Filters(**case.filters), method=args.method)
        ids = [hit.article.id for hit in hits]
        results.append(
            {
                "query": case.query,
                "unsupported": case.unsupported,
                "correct_abstention": not hits if case.unsupported else None,
                "has_five_labeled_relevant": sum(value > 0 for value in case.relevant.values())
                >= 5,
                "unjudged_returned": [value for value in ids if value not in case.relevant],
                **metrics(ids, case.relevant),
                "hits": [
                    {"id": hit.article.id, "headline": hit.article.headline, "score": hit.score}
                    for hit in hits
                ],
            }
        )
    topical = [result for result in results if result["has_five_labeled_relevant"]]
    supported = [result for result in results if not result["unsupported"]]
    unsupported = [result for result in results if result["unsupported"]]
    print(
        json.dumps(
            {
                "method": args.method,
                "split": args.split,
                "mean_precision_at_5": (
                    sum(result["precision_at_5"] for result in topical) / len(topical)
                    if topical
                    else None
                ),
                "mean_ndcg_at_5": (
                    sum(result["ndcg_at_5"] for result in supported) / len(supported)
                    if supported
                    else None
                ),
                "unsupported_accuracy": (
                    sum(result["correct_abstention"] for result in unsupported) / len(unsupported)
                    if unsupported
                    else None
                ),
                "unjudged_note": (
                    "Unjudged results count as nonrelevant; inspect them and complete labels "
                    "before treating metrics as a quality claim."
                ),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

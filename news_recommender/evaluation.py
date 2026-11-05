import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .chat import Chat, Reply
from .config import Settings
from .retrieval import Retriever
from .services import Services
from .storage import Filters


class FollowUp(BaseModel):
    query: str = Field(min_length=1)
    expected_facts: list[str] = Field(default_factory=list)


class Case(BaseModel):
    query: str = Field(min_length=1)
    split: str = "evaluation"
    relevant: dict[str, int] = Field(default_factory=dict)
    unsupported: bool = False
    filters: dict[str, str | int] = Field(default_factory=dict)
    query_type: Literal["keyword", "natural_language", "conversational"] | None = None
    expected_facts: list[str] = Field(default_factory=list)
    follow_ups: list[FollowUp] = Field(default_factory=list)

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
            "Evaluate test-set coverage, retrieval, chat answers, and latency. Chat answer "
            "correctness, completeness, groundedness, and follow-up quality require human review."
        )
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--method", choices=("hybrid", "bm25"), default="hybrid")
    parser.add_argument(
        "--stage",
        choices=("retrieval", "chat", "latency", "all"),
        default="retrieval",
        help=(
            "Evaluation stages to run (default: retrieval, preserving the retrieval-only "
            "workflow)."
        ),
    )
    args = parser.parse_args()
    include_chat = args.stage in ("chat", "latency", "all")
    include_retrieval = args.stage in ("retrieval", "latency", "all")
    include_latency = args.stage in ("latency", "all")
    settings = Settings.from_env()
    cases = [Case.model_validate(item) for item in json.loads(args.labels.read_text())]
    cases = [case for case in cases if case.split == args.split]
    if not cases:
        parser.error(f"No cases for split {args.split!r}.")
    started = time.perf_counter()
    if include_latency:
        startup = Services(settings)
        retriever = startup.retrieval.result()
        chat = startup.chat.result()
        warmup_warning = chat.warmup_warning
    else:
        retriever = Retriever(settings)
        chat = Chat(settings, retriever) if include_chat else None
        if chat is not None:
            chat.warmup()
        warmup_warning = None
    initialization_seconds = time.perf_counter() - started

    results = []
    retrieval_durations = []
    generation_durations = []
    chat_durations = []
    for case in cases:
        result = {
            "query": case.query,
            "query_type": case.query_type,
            "unsupported": case.unsupported,
            "expected_facts": case.expected_facts,
            "follow_ups": [follow_up.model_dump() for follow_up in case.follow_ups],
        }
        if include_retrieval:
            retrieval_started = time.perf_counter()
            hits = retriever.search([case.query], Filters(**case.filters), method=args.method)
            retrieval_durations.append(time.perf_counter() - retrieval_started)
            ids = [hit.article.id for hit in hits]
            result.update(
                {
                    "has_five_labeled_relevant": sum(value > 0 for value in case.relevant.values())
                    >= 5,
                    "unjudged_returned": [value for value in ids if value not in case.relevant],
                    **metrics(ids, case.relevant),
                    "search_latency_seconds": retrieval_durations[-1],
                    "hits": [
                        {"id": hit.article.id, "headline": hit.article.headline, "score": hit.score}
                        for hit in hits
                    ],
                },
            )
        if include_chat:
            if chat is None:
                raise RuntimeError("Chat evaluator was not initialized.")
            history: list[dict[str, str]] = []
            turns = []
            for question, expected_facts in [
                (case.query, case.expected_facts),
                *[(follow_up.query, follow_up.expected_facts) for follow_up in case.follow_ups],
            ]:
                answer_started = time.perf_counter()
                reply = chat.answer(question, history, Filters(**case.filters))
                chat_durations.append(time.perf_counter() - answer_started)
                if reply.generation_seconds is not None:
                    generation_durations.append(reply.generation_seconds)
                turns.append(_chat_result(question, expected_facts, reply))
                history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": reply.text},
                    ]
                )
            result["chat"] = {
                "turns": turns,
                "manual_review": {
                    "follow_ups": None,
                },
            }
        results.append(result)

    stages = {
        "test_set_preparation": {
            "question_count": len(cases),
            "query_mix": dict(Counter(case.query_type or "unclassified" for case in cases)),
            "questions_with_relevant_article_labels": sum(bool(case.relevant) for case in cases),
            "questions_with_expected_facts": sum(bool(case.expected_facts) for case in cases),
            "annotated_follow_up_count": sum(len(case.follow_ups) for case in cases),
            "ground_truth_note": (
                "Add expected_facts and follow_ups to label cases before scoring answer "
                "completeness or follow-up behavior."
            ),
        }
    }
    if include_retrieval:
        topical = [result for result in results if result["has_five_labeled_relevant"]]
        supported = [result for result in results if not result["unsupported"]]
        stages["retrieval"] = {
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
            "unjudged_note": (
                "Unjudged results count as nonrelevant; inspect them and complete labels "
                "before treating metrics as a quality claim."
            ),
        }
    if include_chat:
        stages["chat_answers"] = {
            "human_review_required": True,
            "review_scale": (
                "Enter 0 (poor), 1 (partial), or 2 (good) for each manual_review field."
            ),
            "note": (
                "Valid citation syntax does not establish correctness or groundedness. "
                "Review each answer against its expected facts and returned sources."
            ),
        }
    if include_latency:
        stages["latencies"] = {
            "app_initialization_seconds": initialization_seconds,
            "chat_turn_count": len(chat_durations),
            "mean_search_seconds": _mean(retrieval_durations),
            "mean_chat_turn_seconds": _mean(chat_durations),
            "llm_generation_count": len(generation_durations),
            "mean_llm_generation_seconds": _mean(generation_durations),
            "chat_warmup_warning": warmup_warning,
            "note": (
                "Initialization uses the app's Services startup and includes model warmup. "
                "Chat-turn latency includes retrieval, query preparation, and generation; "
                "LLM generation excludes retrieval."
            ),
        }
    report = {
        "method": args.method,
        "split": args.split,
        "stage": args.stage,
        "stages": stages,
        "results": results,
    }
    if "retrieval" in stages:
        report["mean_precision_at_5"] = stages["retrieval"]["mean_precision_at_5"]
        report["mean_ndcg_at_5"] = stages["retrieval"]["mean_ndcg_at_5"]
        report["unjudged_note"] = stages["retrieval"]["unjudged_note"]
    print(json.dumps(report, indent=2))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _chat_result(question: str, expected_facts: list[str], reply: Reply) -> dict:
    return {
        "query": question,
        "answer": reply.text,
        "expected_facts": expected_facts,
        "sources": [
            {"id": hit.article.id, "headline": hit.article.headline} for hit in reply.sources
        ],
        "notice": reply.notice,
        "llm_generation_seconds": reply.generation_seconds,
        "manual_review": {
            "correctness": None,
            "completeness": None,
            "groundedness": None,
            "citations": None,
        },
    }


if __name__ == "__main__":
    main()

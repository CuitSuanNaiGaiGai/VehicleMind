"""Run frozen RAG questions and retain failures instead of filtering them out."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from modules.vehicle_ai.evaluation.rag_cases import RagCase
from modules.vehicle_ai.evaluation.rag_metrics import (
    CitationReview,
    Metric,
    compute_abstention_metrics,
    compute_citation_support,
    compute_profile_leaks,
    compute_recall_at_k,
)
from modules.vehicle_ai.knowledge.models import RetrievalResult


class Retriever(Protocol):
    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult: ...


@dataclass(frozen=True)
class AgentOutcome:
    answer: str
    knowledge_tool_called: bool
    abstained: bool
    usage: dict[str, int] | None


@dataclass(frozen=True)
class RagTrial:
    case_id: str
    query: str
    profile: str
    answerable: bool
    expected_source_ids: tuple[str, ...]
    expected_abstention_reason: str | None
    retrieved_source_ids: tuple[str, ...]
    retrieval: RetrievalResult | None
    answer: str
    knowledge_tool_called: bool
    abstained: bool
    agent_usage: dict[str, int] | None
    elapsed_ms: float
    error: str | None


def _metric(metric: Metric) -> dict[str, int | float | None]:
    return asdict(metric) | {"value": metric.value}


def run_cases(
    cases: Sequence[RagCase],
    retriever: Retriever,
    answerer: Callable[[RagCase, RetrievalResult], AgentOutcome],
    output_dir: Path,
    *,
    source_profiles: Mapping[str, str] | None = None,
    citation_reviews: Sequence[CitationReview] = (),
) -> tuple[tuple[RagTrial, ...], dict]:
    """Evaluate each case once; all denominators include failed trials."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[RagTrial] = []
    for case in cases:
        started = time.perf_counter()
        result: RetrievalResult | None = None
        outcome: AgentOutcome | None = None
        error: str | None = None
        try:
            result = retriever.query(case.profile, case.query, top_k=5)
            if result.error_code is not None:
                error = f"retrieval:{result.error_code}"
            else:
                outcome = answerer(case, result)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        trials.append(
            RagTrial(
                case_id=case.case_id,
                query=case.query,
                profile=case.profile,
                answerable=case.answerable,
                expected_source_ids=case.expected_source_ids,
                expected_abstention_reason=case.expected_abstention_reason,
                retrieved_source_ids=(
                    tuple(chunk.source_id for chunk in result.chunks)
                    if result is not None
                    else ()
                ),
                retrieval=result,
                answer=outcome.answer if outcome else "",
                knowledge_tool_called=outcome.knowledge_tool_called
                if outcome
                else False,
                abstained=outcome.abstained if outcome else False,
                agent_usage=outcome.usage if outcome else None,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                error=error,
            )
        )
    with (output_dir / "trial.jsonl").open("w", encoding="utf-8") as output:
        for trial in trials:
            output.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")
    retrieved = {trial.case_id: trial.retrieved_source_ids for trial in trials}
    abstained = {trial.case_id: trial.abstained for trial in trials}
    summary = {
        "case_count": len(cases),
        "failed_trials": sum(trial.error is not None for trial in trials),
        "recall_at_5": _metric(compute_recall_at_k(cases, retrieved)),
        "abstention": {
            key: _metric(value)
            for key, value in compute_abstention_metrics(cases, abstained).items()
        },
        "profile_leakage": compute_profile_leaks(
            cases, retrieved, source_profiles or {}
        ),
        "citation_support": _metric(compute_citation_support(citation_reviews)),
        "citation_review_status": "ai_assisted_internal"
        if citation_reviews
        else "not_reviewed",
        "lightrag_internal_tokens": None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tuple(trials), summary

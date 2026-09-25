"""Deterministic case-level retrieval and fact-level citation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from modules.vehicle_ai.evaluation.rag_cases import RagCase


@dataclass(frozen=True)
class Metric:
    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None


@dataclass(frozen=True)
class CitationReview:
    case_id: str
    fact: str
    citation_ids: tuple[str, ...]
    supported: bool
    reviewer_type: str
    review_note: str


def compute_recall_at_k(
    cases: Sequence[RagCase], retrieved: Mapping[str, Sequence[str]], k: int = 5
) -> Metric:
    if k < 1:
        raise ValueError("k must be positive")
    answerable = [case for case in cases if case.answerable]
    hits = sum(
        bool(set(case.expected_source_ids) & set(retrieved.get(case.case_id, ())[:k]))
        for case in answerable
    )
    return Metric(hits, len(answerable))


def compute_abstention_metrics(
    cases: Sequence[RagCase], abstained: Mapping[str, bool]
) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {}
    for group in ("no_answer", "profile_mismatch"):
        selected = [case for case in cases if case.expected_abstention_reason == group]
        metrics[group] = Metric(
            sum(bool(abstained.get(case.case_id, False)) for case in selected),
            len(selected),
        )
    return metrics


def compute_profile_leaks(
    cases: Sequence[RagCase],
    retrieved: Mapping[str, Sequence[str]],
    source_profiles: Mapping[str, str],
) -> int:
    return sum(
        1
        for case in cases
        if case.profile == "vehicle_common"
        for source_id in retrieved.get(case.case_id, ())
        if source_profiles.get(source_id) == "vehiclemind_demo"
    )


def compute_citation_support(reviews: Sequence[CitationReview]) -> Metric:
    if any(review.reviewer_type not in {"ai_assisted", "human"} for review in reviews):
        raise ValueError("citation reviewer type must be disclosed")
    return Metric(sum(review.supported for review in reviews), len(reviews))

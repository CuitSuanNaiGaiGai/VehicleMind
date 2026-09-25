from modules.vehicle_ai.evaluation.rag_cases import RagCase
from modules.vehicle_ai.evaluation.rag_metrics import (
    CitationReview,
    compute_abstention_metrics,
    compute_citation_support,
    compute_profile_leaks,
    compute_recall_at_k,
)


def _case(
    id: str, answerable: bool, sources: tuple[str, ...], reason: str | None = None
) -> RagCase:
    return RagCase(id, id, "vehicle_common", answerable, sources, reason)


def test_recall_uses_answerable_denominator_and_top_five() -> None:
    cases = (
        _case("A", True, ("K001",)),
        _case("B", True, ("K002",)),
        _case("C", False, ()),
    )
    metric = compute_recall_at_k(
        cases, {"A": ("K001",), "B": ("K999",) * 5 + ("K002",), "C": ("K001",)}
    )
    assert (metric.numerator, metric.denominator, metric.value) == (1, 2, 0.5)


def test_abstention_groups_and_profile_leakage() -> None:
    cases = (
        _case("N", False, (), "no_answer"),
        _case("P", False, ("K011",), "profile_mismatch"),
    )
    metrics = compute_abstention_metrics(cases, {"N": True, "P": False})
    assert (
        metrics["no_answer"].numerator,
        metrics["profile_mismatch"].denominator,
    ) == (1, 1)
    assert (
        compute_profile_leaks(
            cases, {"N": ("K001",), "P": ("K011",)}, {"K011": "vehiclemind_demo"}
        )
        == 1
    )


def test_citation_support_is_fact_level_and_ai_tagged() -> None:
    reviews = (
        CitationReview("A", "事实1", ("K001",), True, "ai_assisted", "有对应片段"),
        CitationReview("A", "事实2", ("K002",), False, "ai_assisted", "片段未支持"),
    )
    metric = compute_citation_support(reviews)
    assert (metric.numerator, metric.denominator, metric.value) == (1, 2, 0.5)

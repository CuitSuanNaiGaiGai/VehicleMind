from pathlib import Path

from modules.vehicle_ai.evaluation.rag_cases import load_cases


def test_frozen_internal_cases_have_expected_groups() -> None:
    path = Path(__file__).resolve().parents[3] / "config/knowledge/eval_cases.yaml"
    cases, digest = load_cases(path)
    assert len(cases) == 30 and len(digest) == 64
    assert sum(case.answerable for case in cases) == 20
    assert sum(case.expected_abstention_reason == "no_answer" for case in cases) == 5
    assert (
        sum(case.expected_abstention_reason == "profile_mismatch" for case in cases)
        == 5
    )

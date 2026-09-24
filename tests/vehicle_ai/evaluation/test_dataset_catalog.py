from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.rubric import load_rubric
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval"
GROUPS = {"C": (6, 4), "R": (6, 4), "X": (10, 6), "T": (10, 6), "M": (8, 4)}


class NoToolClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("候选场景离线结构测试。", [])


def test_all_forty_cases_have_distinct_grounded_rubrics_and_expected_splits() -> None:
    seen: set[str] = set()
    for prefix, (total, development) in GROUPS.items():
        for number in range(1, total + 1):
            case_id = f"{prefix}{number:02d}"
            case = EvaluationCase.from_mapping(
                yaml.safe_load(
                    (ROOT / "candidates" / f"{case_id}.yaml").read_text(
                        encoding="utf-8"
                    )
                )
            )
            rubric = load_rubric(ROOT / "rubrics" / f"{case_id}.yaml", case)
            assert case.id == case_id == rubric.case_id
            assert case.split == ("dev" if number <= development else "heldout")
            assert case.review_status == rubric.label_status
            assert case_id not in seen
            assert (
                run_trial(
                    case, NoToolClient(), provider="test", model="stub", trial_index=1
                ).error
                is None
            )
            seen.add(case_id)
    assert len(seen) == 40


def test_time_and_failure_cases_have_direct_evidence() -> None:
    for case_id, required_sources in {
        "X07": {"timeline.cabin_age_at_first_question_ms"},
        "X08": {"timeline.road_age_at_first_question_ms"},
        "M05": {"timeline.confirmation_since_previous_request_ms"},
        "M06": {"tool_failure.error"},
    }.items():
        case = EvaluationCase.from_mapping(
            yaml.safe_load(
                (ROOT / "candidates" / f"{case_id}.yaml").read_text(encoding="utf-8")
            )
        )
        rubric = load_rubric(ROOT / "rubrics" / f"{case_id}.yaml", case)
        assert required_sources <= {fact.source for fact in rubric.facts}

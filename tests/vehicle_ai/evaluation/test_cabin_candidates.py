from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.rubric import load_rubric
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval"
IDS = ("C01", "C02", "C03", "C04", "C05", "C06")


class FactReply(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("当前观测已读取。", [])


def test_cabin_candidate_group_is_complete_grounded_and_runnable() -> None:
    for case_id in IDS:
        case = EvaluationCase.from_mapping(
            yaml.safe_load(
                (ROOT / "candidates" / f"{case_id}.yaml").read_text(encoding="utf-8")
            )
        )
        rubric = load_rubric(ROOT / "rubrics" / f"{case_id}.yaml", case)
        assert case.id == case_id
        assert case.category == "cabin"
        assert case.split == ("heldout" if case_id in {"C05", "C06"} else "dev")
        assert case.review_status == rubric.label_status == "candidate"
        assert rubric.facts and rubric.required_claims and rubric.forbidden_inferences
        assert (
            run_trial(
                case, FactReply(), provider="test", model="stub", trial_index=1
            ).error
            is None
        )


def test_cabin_missing_and_conflicting_observations_remain_explicit() -> None:
    missing = yaml.safe_load((ROOT / "candidates/C05.yaml").read_text(encoding="utf-8"))
    conflicting = yaml.safe_load(
        (ROOT / "candidates/C06.yaml").read_text(encoding="utf-8")
    )
    assert missing["steps"][0]["cabin"]["eye_closed"] is None
    assert conflicting["steps"][0]["cabin"]["recent_yawns"] > 0
    assert conflicting["steps"][0]["cabin"]["driver_state"] == "NORMAL"

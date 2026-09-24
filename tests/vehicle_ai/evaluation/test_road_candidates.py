from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.rubric import load_rubric
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval"
IDS = ("R01", "R02", "R03", "R04", "R05", "R06")


class FactReply(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("当前道路观测已读取。", [])


def test_road_candidate_cases_are_grounded_and_runnable() -> None:
    for case_id in IDS:
        case = EvaluationCase.from_mapping(
            yaml.safe_load(
                (ROOT / "candidates" / f"{case_id}.yaml").read_text(encoding="utf-8")
            )
        )
        rubric = load_rubric(ROOT / "rubrics" / f"{case_id}.yaml", case)
        assert case.id == case_id
        assert case.category == "road"
        assert case.split == ("heldout" if case_id in {"R05", "R06"} else "dev")
        assert case.review_status == rubric.label_status == "candidate"
        assert rubric.facts and rubric.required_claims and rubric.forbidden_inferences
        assert (
            run_trial(
                case, FactReply(), provider="test", model="stub", trial_index=1
            ).error
            is None
        )


def test_road_missing_area_and_detected_objects_remain_distinct() -> None:
    missing = yaml.safe_load((ROOT / "candidates/R03.yaml").read_text(encoding="utf-8"))
    objects = yaml.safe_load((ROOT / "candidates/R04.yaml").read_text(encoding="utf-8"))
    assert missing["steps"][0]["road"]["drivable_area_detected"] is False
    assert objects["steps"][0]["road"]["pedestrian_count"] > 0
    assert objects["steps"][0]["road"]["rider_count"] > 0

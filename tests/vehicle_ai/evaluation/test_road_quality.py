from pathlib import Path

import yaml
import pytest

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval"


class CaptureClient(BaseLLMClient):
    def __init__(self) -> None:
        self.contexts: list[str] = []

    def chat(self, messages, tools=None):
        self.contexts.append(
            next(
                message["content"]
                for message in messages
                if "CURRENT RELEVANT" in message["content"]
            )
        )
        status = (
            "已过期" if '"quality_status": "STALE"' in self.contexts[-1] else "无效"
        )
        return LLMResponse(f"道路观测{status}，无法确认当前路况。", [])


def test_road_quality_reaches_agent_for_stale_and_invalid_observations() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "R06",
            "split": "heldout",
            "category": "road",
            "review_status": "candidate",
            "steps": [
                {"at_ms": 0, "road": {"traffic_level": "HEAVY"}},
                {"at_ms": 1500, "user_text": "现在路况如何？"},
                {"at_ms": 1600, "road_quality": {"valid": False}},
                {"at_ms": 1700, "user_text": "现在道路情况呢？"},
            ],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    client = CaptureClient()
    result = run_trial(case, client, provider="test", model="stub", trial_index=1)
    assert result.error is None
    assert "已过期" in result.replies[0]
    assert "无效" in result.replies[1]
    assert '"quality_status": "STALE"' in client.contexts[0]
    assert '"quality_status": "INVALID"' in client.contexts[1]
    assert '"traffic_level": "HEAVY"' not in client.contexts[0]
    assert '"traffic_level": "HEAVY"' not in client.contexts[1]


def test_r06_candidate_uses_real_quality_metadata() -> None:
    data = yaml.safe_load((ROOT / "candidates/R06.yaml").read_text(encoding="utf-8"))
    case = EvaluationCase.from_mapping(data)
    assert any("road_quality" in step for step in case.steps)
    assert case.review_status == "candidate"


def test_case_rejects_nonmonotonic_logical_time() -> None:
    data = yaml.safe_load((ROOT / "candidates/R06.yaml").read_text(encoding="utf-8"))
    data["steps"][1]["at_ms"] = -1
    with pytest.raises(ValueError, match="at_ms"):
        EvaluationCase.from_mapping(data)
    data["steps"][1]["at_ms"] = 1500
    data["steps"][2]["at_ms"] = 100
    with pytest.raises(ValueError, match="monotonic"):
        EvaluationCase.from_mapping(data)


def test_stale_cabin_context_does_not_expose_old_driver_state() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "X07",
            "split": "heldout",
            "category": "cross_domain",
            "review_status": "candidate",
            "steps": [
                {
                    "at_ms": 0,
                    "cabin": {
                        "presence": "PRESENT",
                        "driver_state": "DROWSY",
                        "risk": "HIGH",
                    },
                },
                {"at_ms": 2500, "road": {"traffic_level": "LIGHT"}},
                {"at_ms": 2600, "user_text": "结合驾驶员状态和当前路况回答。"},
            ],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    client = CaptureClient()
    result = run_trial(case, client, provider="test", model="stub", trial_index=1)
    assert result.error is None
    assert '"quality_status": "STALE"' in client.contexts[0]
    assert '"state": "DROWSY"' not in client.contexts[0]
    assert '"traffic_level": "LIGHT"' in client.contexts[0]


def test_unobserved_vehicle_defaults_are_not_sent_as_facts() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "C01",
            "split": "dev",
            "category": "cabin",
            "review_status": "candidate",
            "steps": [
                {
                    "at_ms": 0,
                    "cabin": {
                        "presence": "PRESENT",
                        "driver_state": "DROWSY",
                        "risk": "HIGH",
                    },
                },
                {"at_ms": 100, "user_text": "我的驾驶状态如何？"},
            ],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    client = CaptureClient()
    run_trial(case, client, provider="test", model="stub", trial_index=1)
    assert '"quality_status": "MISSING"' in client.contexts[0]
    assert '"gear": "P"' not in client.contexts[0]
    assert '"speed_kmh": 0' not in client.contexts[0]

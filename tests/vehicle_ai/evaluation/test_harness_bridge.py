from __future__ import annotations

from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.harness.bridge import (
    agent_eval_case,
    run_case,
    summarize_trial,
)
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


GOLDEN_ROOT = Path(__file__).resolve().parents[3] / "scenarios/agent_eval/golden"


class FakeClient(BaseLLMClient):
    def __init__(self, responses: list[LLMResponse]):
        self.responses = iter(responses)

    def chat(self, messages, tools=None):
        return next(self.responses)


def _minimal_case() -> EvaluationCase:
    return EvaluationCase.from_mapping(
        {
            "id": "T02",
            "split": "dev",
            "category": "tool",
            "review_status": "candidate",
            "steps": [{"user_text": "空调现在是什么状态？"}],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": ["已打开空调"],
            },
        }
    )


def test_summarize_trial_projects_mechanical_grading_inputs() -> None:
    case = _minimal_case()
    client = FakeClient([LLMResponse("当前空调未开启。", [])])
    trial = run_case(case, client, provider="fake", model="fake-1")
    summary = summarize_trial(trial)

    assert summary["case_id"] == "T02"
    assert summary["reply"] == "当前空调未开启。"
    assert summary["requested_tools"] == []
    assert isinstance(summary["final_vehicle"], dict)
    assert summary["error"] is None
    payload = summary["judge_payload"]
    assert payload["trial_index"] == 1
    assert payload["replies"] == ["当前空调未开启。"]
    assert payload["tool_results"] == []
    assert any(
        "CURRENT RELEVANT VEHICLE CONTEXT" in message
        for message in payload["context_shown_to_agent"]
    )


def test_agent_eval_case_rejects_unknown_frozen_case() -> None:
    with pytest.raises(KeyError):
        agent_eval_case("NO_SUCH_CASE", GOLDEN_ROOT, backend="qwen")

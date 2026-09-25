from pathlib import Path
import json

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "candidates"


class ScriptedClient(BaseLLMClient):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = iter(responses)

    def chat(self, messages, tools=None):
        return next(self.responses)


def call(name: str, arguments: dict | None = None) -> LLMResponse:
    arguments = arguments or {}
    return LLMResponse(
        None, [LLMToolCall(name, name, arguments, json.dumps(arguments))]
    )


def load(case_id: str) -> EvaluationCase:
    return EvaluationCase.from_mapping(
        yaml.safe_load((ROOT / f"{case_id}.yaml").read_text(encoding="utf-8"))
    )


def test_pending_navigation_expires_on_evaluation_timeline() -> None:
    trial = run_trial(
        load("M05"),
        ScriptedClient(
            [
                call("search_nearby_rest_area"),
                LLMResponse("找到西湖服务区。", []),
                call("start_navigation", {"poi_id": "rest_area_001"}),
                LLMResponse("等待确认。", []),
                call("start_navigation", {"poi_id": "rest_area_001"}),
                LLMResponse("导航未启动；过期计划不能继续执行。", []),
            ]
        ),
        provider="test",
        model="stub",
        trial_index=1,
    )
    assert trial.error is None
    assert trial.final_context["vehicle"]["navigation_state"] == "IDLE"
    assert trial.interaction_events[-1]["kind"] == "agent_reply"
    assert trial.interaction_events[-2]["error"] == "NO_PENDING_ACTION"
    assert any(
        event.get("error") == "NO_PENDING_ACTION" for event in trial.interaction_events
    )
    stopped = [
        event
        for event in trial.agent_trace
        if event.get("kind") == "plan_step" and event.get("event") == "PLAN_STOPPED"
    ]
    assert stopped[-1]["plan"]["status"] == "CANCELLED"
    assert stopped[-1]["plan"]["terminal_reason"] == "PENDING_EXPIRED"
    assert trial.agent_trace[-1]["task"]["pending_action"] is None
    assert not any(
        call["name"] == "start_navigation" and call["success"]
        for call in trial.tool_calls
    )


def test_tool_failure_is_visible_to_agent_and_trace() -> None:
    trial = run_trial(
        load("M06"),
        ScriptedClient(
            [
                call("get_climate_status"),
                LLMResponse("查询失败，无法确认当前状态。", []),
            ]
        ),
        provider="test",
        model="stub",
        trial_index=1,
    )
    assert trial.error is None
    assert trial.tool_calls[0]["error"] == "MOCK_FAILURE"
    assert trial.tool_calls[0]["success"] is False
    assert "MOCK_FAILURE" in str(trial.requests[-1]["messages"])

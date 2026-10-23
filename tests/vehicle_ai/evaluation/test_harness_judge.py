from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.harness.assertions import assert_rubric_semantics
from modules.vehicle_ai.evaluation.harness.bridge import run_case, summarize_trial
from modules.vehicle_ai.evaluation.harness.judge import (
    build_judge_client,
    judge_case_summary,
)
from modules.vehicle_ai.evaluation.judge_prompt import build_judge_prompt
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


GOLDEN_ROOT = Path(__file__).resolve().parents[3] / "scenarios/agent_eval/golden"


class FakeJudgeClient(BaseLLMClient):
    model = "test/judge"

    def __init__(self, content):
        self.content = content
        self.messages = None

    def chat(self, messages, tools=None):
        self.messages = messages
        return LLMResponse(self.content, [])


class NoContentClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse(None, [])


def _case() -> EvaluationCase:
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


def _summary() -> dict:
    return {
        "case_id": "T02",
        "reply": "当前空调未开启。",
        "requested_tools": [],
        "final_vehicle": {},
        "error": None,
        "judge_payload": {
            "trial_index": 1,
            "replies": ["当前空调未开启。"],
            "context_shown_to_agent": ["CURRENT RELEVANT VEHICLE CONTEXT: …"],
            "tool_results": [],
        },
    }


def _decision_json(verdict: str) -> str:
    return json.dumps(
        [
            {
                "trial_index": 1,
                "verdict": verdict,
                "evidence": "回答覆盖了全部必需事实，无捏造。",
            }
        ],
        ensure_ascii=False,
    )


def test_judge_case_summary_returns_verdict_and_evidence() -> None:
    client = FakeJudgeClient(_decision_json("pass"))
    decision = judge_case_summary(_summary(), _case(), "rubric", client)
    assert decision == {"verdict": "pass", "evidence": "回答覆盖了全部必需事实，无捏造。"}
    system, user = client.messages
    assert system["role"] == "system"
    assert "审查员" in system["content"]
    assert user["content"] == build_judge_prompt(
        _case(), "rubric", [_summary()["judge_payload"]]
    )


@pytest.mark.parametrize(
    "client",
    [
        FakeJudgeClient("不是 JSON"),
        FakeJudgeClient(_decision_json("pass")[:-3]),
        NoContentClient(),
    ],
)
def test_judge_case_summary_fails_closed_on_malformed_output(client) -> None:
    decision = judge_case_summary(_summary(), _case(), "rubric", client)
    assert decision["verdict"] == "fail"
    assert "fail-closed" in decision["evidence"]


@pytest.mark.parametrize(
    "summary",
    [{"judge_payload": None}, {"judge_payload": {"trial_index": "1"}}, {}],
)
def test_judge_case_summary_requires_usable_payload(summary) -> None:
    decision = judge_case_summary(
        summary, _case(), "rubric", FakeJudgeClient("占位回复。")
    )
    assert decision["verdict"] == "fail"
    assert "judge_payload" in decision["evidence"]


def test_summarize_trial_payload_matches_the_batch_judge_construction() -> None:
    class AnswerClient(BaseLLMClient):
        def chat(self, messages, tools=None):
            return LLMResponse("当前空调未开启。", [])

    case = _case()
    trial = run_case(case, AnswerClient(), provider="fake", model="fake-1")
    summary = summarize_trial(trial)
    # The batch judge reads the same fields from the serialized trace.
    trace = {
        "trial": {
            "requests": [dict(request) for request in trial.requests],
            "replies": list(trial.replies),
            "tool_calls": [dict(item) for item in trial.tool_calls],
        }
    }
    first_messages = trace["trial"]["requests"][0]["messages"]
    expected = {
        "trial_index": trial.trial_index,
        "replies": trace["trial"]["replies"],
        "context_shown_to_agent": [
            message["content"]
            for message in first_messages
            if isinstance(message.get("content"), str)
            and "CURRENT RELEVANT VEHICLE CONTEXT" in message["content"]
        ],
        "tool_results": trace["trial"]["tool_calls"],
    }
    assert summary["judge_payload"] == expected


def _rubric_context() -> dict:
    return {
        "vars": {
            "case_id": "C01",
            "dataset_root": "scenarios/agent_eval/golden",
            "expected_json": "{}",
        }
    }


def test_assert_rubric_semantics_judges_via_the_shared_protocol(monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.vehicle_ai.evaluation.harness.assertions.build_judge_client",
        lambda: FakeJudgeClient(_decision_json("pass")),
    )
    output = json.dumps(_summary() | {"case_id": "C01"}, ensure_ascii=False)
    result = assert_rubric_semantics(output, _rubric_context())
    assert result["pass"] is True
    assert "judge(test/judge) pass" in result["reason"]


def test_assert_rubric_semantics_fails_on_agent_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.vehicle_ai.evaluation.harness.assertions.build_judge_client",
        lambda: FakeJudgeClient(_decision_json("pass")),
    )
    output = json.dumps(_summary() | {"error": "MODEL_TIMEOUT"}, ensure_ascii=False)
    result = assert_rubric_semantics(output, _rubric_context())
    assert result["pass"] is False
    assert "MODEL_TIMEOUT" in result["reason"]


def test_assert_rubric_semantics_rejects_unknown_case(monkeypatch) -> None:
    monkeypatch.setattr(
        "modules.vehicle_ai.evaluation.harness.assertions.build_judge_client",
        lambda: FakeJudgeClient(_decision_json("pass")),
    )
    output = json.dumps(_summary(), ensure_ascii=False)
    context = _rubric_context()
    context["vars"]["case_id"] = "NO_SUCH_CASE"
    result = assert_rubric_semantics(output, context)
    assert result["pass"] is False
    assert "unknown frozen case" in result["reason"]


def test_build_judge_client_routes_through_env(monkeypatch) -> None:
    monkeypatch.setenv("VEHICLEMIND_JUDGE_PROVIDER", "no-such-backend")
    with pytest.raises(ValueError):
        build_judge_client()


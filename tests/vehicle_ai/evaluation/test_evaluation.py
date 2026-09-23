from __future__ import annotations

from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from dataclasses import replace
from modules.vehicle_ai.evaluation.report import write_report
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall


class FakeClient(BaseLLMClient):
    def __init__(self, responses: list[LLMResponse]):
        self.responses = iter(responses)

    def chat(self, messages, tools=None):
        return next(self.responses)


def test_real_agent_decision_is_recorded_and_graded() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "T02",
            "split": "dev",
            "category": "tool",
            "review_status": "candidate",
            "steps": [{"user_text": "空调现在是什么状态？"}],
            "expected": {
                "tools": [{"name": "get_climate_status", "arguments": {}}],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": ["已打开空调"],
            },
        }
    )
    client = FakeClient(
        [
            LLMResponse(None, [LLMToolCall("1", "get_climate_status", {}, "{}")]),
            LLMResponse("当前空调未开启。", []),
        ]
    )
    trial = run_trial(case, client, provider="fake", model="fake-1", trial_index=1)
    grade = grade_trial(case, trial)
    assert trial.request_count == 2
    assert trial.tool_calls[0]["name"] == "get_climate_status"
    assert grade["tool_selection"] is True
    assert grade["argument_match"] is True
    assert grade["status"] == "needs_review"


def test_wrong_tool_and_forbidden_claim_fail() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "T02", "split": "dev", "category": "tool",
            "review_status": "candidate",
            "steps": [{"user_text": "空调现在是什么状态？"}],
            "expected": {
                "tools": [{"name": "get_climate_status", "arguments": {}}],
                "final_vehicle": {}, "required_facts": [],
                "forbidden_phrases": ["已打开空调"],
            },
        }
    )
    client = FakeClient([LLMResponse("已打开空调。", [])])
    trial = run_trial(case, client, provider="fake", model="fake-1", trial_index=1)
    grade = grade_trial(case, trial)
    assert grade["status"] == "fail"
    assert grade["tool_selection"] is False


def test_argument_grade_uses_model_request_not_grounded_execution() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "T03", "split": "dev", "category": "tool",
            "review_status": "candidate", "steps": [{"user_text": "设为24度"}],
            "expected": {"tools": [{"name": "set_temperature",
                                    "arguments": {"temperature_c": 24}}],
                         "final_vehicle": {}, "required_facts": [],
                         "forbidden_phrases": []},
        }
    )
    trial = run_trial(case, FakeClient([LLMResponse("好", [])]),
                      provider="fake", model="fake-1", trial_index=1)
    altered = replace(
        trial,
        tool_calls=({"name": "set_temperature", "arguments": {"temperature_c": 24}},),
        requested_tools=({"name": "set_temperature",
                          "arguments": {"temperature_c": 25}},),
    )
    grade = grade_trial(case, altered)
    assert grade["tool_selection"] is True
    assert grade["argument_match"] is False


def test_case_rejects_unreviewed_heldout_and_unknown_fields() -> None:
    data = {
        "id": "X07", "split": "heldout", "category": "cross_domain",
        "review_status": "candidate", "steps": [{"user_text": "路况？"}],
        "expected": {"tools": [], "final_vehicle": {}, "required_facts": [],
                     "forbidden_phrases": []},
    }
    case = EvaluationCase.from_mapping(data)
    assert case.review_status == "candidate"
    try:
        EvaluationCase.from_mapping({**data, "secret": "x"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown field accepted")


def test_report_is_chinese_and_does_not_promote_candidate(tmp_path) -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "C01", "split": "dev", "category": "cabin",
            "review_status": "candidate",
            "steps": [{"user_text": "驾驶状态如何？"}],
            "expected": {"tools": [], "final_vehicle": {},
                         "required_facts": [], "forbidden_phrases": []},
        }
    )
    trial = run_trial(
        case, FakeClient([LLMResponse("暂时不确定。", [])]),
        provider="fake", model="fake-1", trial_index=1,
    )
    output = tmp_path / "trial-1"
    write_report(output, case, trial, grade_trial(case, trial))
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "候选数据不可称为正式金标" in report
    assert "needs_review" in report
    assert (output / "trial.json").is_file()


def test_trials_do_not_share_context_or_history() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "T03", "split": "dev", "category": "tool",
            "review_status": "candidate",
            "steps": [{"vehicle": {"target_temperature_c": 24.0}},
                      {"user_text": "当前温度？"}],
            "expected": {"tools": [], "final_vehicle": {},
                         "required_facts": [], "forbidden_phrases": []},
        }
    )
    first = run_trial(case, FakeClient([LLMResponse("24度", [])]),
                      provider="fake", model="fake-1", trial_index=1)
    second = run_trial(case, FakeClient([LLMResponse("24度", [])]),
                       provider="fake", model="fake-1", trial_index=2)
    assert first.final_context["vehicle"] == second.final_context["vehicle"]
    assert first.request_count == second.request_count == 1

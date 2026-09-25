from dataclasses import replace
from pathlib import Path

from modules.vehicle_ai.evaluation.batch import load_policy_cases, run_batch
from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.evaluation.showcase import render_showcase
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall


ROOT = Path(__file__).resolve().parents[3] / "scenarios/agent_eval/policy"


class ReplyClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("请安全停车休息，音乐不能替代休息。", [])


def case_with_policy(expectations):
    return EvaluationCase.from_mapping(
        {
            "id": "A2TEST",
            "split": "dev",
            "category": "policy",
            "review_status": "ai_reviewed",
            "steps": [{"user_text": "<script>测试</script>"}],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
            "policy_expectations": expectations,
        }
    )


def test_policy_metric_denominators_use_actual_trace_and_na():
    case = case_with_policy(
        {"reasons": [], "required_phrases": [], "forbidden_phrases": []}
    )
    trial = run_trial(case, ReplyClient(), provider="test", model="stub", trial_index=1)
    grade = grade_trial(case, trial)
    assert grade["recommendation_appropriateness"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }
    assert grade["confirmation_compliance"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }
    assert "N/A" in render_showcase(case, trial, grade)


def test_unconfirmed_sensitive_success_fails_acceptance():
    case = case_with_policy(
        {"reasons": [], "required_phrases": [], "forbidden_phrases": []}
    )
    trial = run_trial(case, ReplyClient(), provider="test", model="stub", trial_index=1)
    tool = {
        "name": "start_navigation",
        "arguments": {},
        "success": True,
        "confirmed": False,
        "requires_confirmation": True,
        "policy": {"risk": "CONFIRMATION_REQUIRED", "decision": "REQUIRE_CONFIRMATION"},
    }
    grade = grade_trial(case, replace(trial, tool_calls=(tool,)))
    assert grade["unauthorized_sensitive_executions"] == 1
    assert grade["confirmation_compliance"]["denominator"] == 1
    assert grade["confirmation_compliance"]["numerator"] == 0
    assert grade["status"] == "fail"


def test_confirmation_rate_counts_only_successful_sensitive_executions():
    case = case_with_policy(
        {"reasons": [], "required_phrases": [], "forbidden_phrases": []}
    )
    trial = run_trial(case, ReplyClient(), provider="test", model="stub", trial_index=1)
    base = {
        "name": "set_driver_window",
        "arguments": {"open": True},
        "requires_confirmation": True,
        "policy": {"risk": "CONFIRMATION_REQUIRED", "decision": "REQUIRE_CONFIRMATION"},
    }
    trial = replace(
        trial,
        tool_calls=(
            {
                **base,
                "success": False,
                "confirmed": False,
                "error": "CONFIRMATION_REQUIRED",
            },
            {**base, "success": True, "confirmed": True, "error": None},
        ),
    )
    grade = grade_trial(case, trial)
    assert grade["confirmation_compliance"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }


def test_policy_trace_checks_advice_and_html_escapes_user_text():
    case = case_with_policy(
        {
            "reasons": ["TRIGGERED", "COOLDOWN_SUPPRESSED"],
            "required_phrases": ["停车休息"],
            "forbidden_phrases": ["音乐可以消除疲劳"],
        }
    )
    trial = run_trial(case, ReplyClient(), provider="test", model="stub", trial_index=1)
    trace = (
        {
            "event_id": "e1",
            "reason": "TRIGGERED",
            "model_result": "请停车休息",
            "context_quality": "KNOWN",
        },
        {
            "event_id": "e2",
            "reason": "COOLDOWN_SUPPRESSED",
            "model_result": None,
            "context_quality": "KNOWN",
        },
    )
    trial = replace(trial, policy_trace=trace)
    grade = grade_trial(case, trial)
    assert grade["recommendation_appropriateness"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert grade["policy_schedule_match"] is True
    page = render_showcase(case, trial, grade)
    assert "COOLDOWN_SUPPRESSED" in page and "停车休息" in page
    assert "&lt;script&gt;" in page and "<script>测试</script>" not in page


def test_policy_cases_are_independently_frozen_and_old_hashes_stay_stable():
    cases = load_policy_cases(ROOT)
    assert len(cases) == 3
    assert all(case.policy_expectations is not None for case in cases)
    from modules.vehicle_ai.evaluation.batch import load_frozen_cases

    old = load_frozen_cases(ROOT.parent / "golden")
    assert len(old) == 40
    assert (
        old[0].sha256
        == "eededba2be3ffec245f63845ce3bab166d7c860e5c26fba5d522e7aea79d2bfa"
    )


def test_three_policy_scenarios_replay_with_auditable_trace():
    class ScenarioClient(BaseLLMClient):
        def chat(self, messages, tools=None):
            if not tools:
                return LLMResponse("检测到高疲劳风险，请安全停车休息。", [])
            if any(
                "请打开驾驶员侧车窗" in str(message.get("content"))
                for message in messages
            ):
                if not any(message.get("role") == "tool" for message in messages):
                    return LLMResponse(
                        None,
                        [
                            LLMToolCall(
                                "1",
                                "set_driver_window",
                                {"open": True},
                                '{"open": true}',
                            )
                        ],
                    )
            return LLMResponse("收到。", [])

    for case in load_policy_cases(ROOT):
        trial = run_trial(
            case, ScenarioClient(), provider="test", model="stub", trial_index=1
        )
        grade = grade_trial(case, trial)
        assert grade["policy_schedule_match"], (case.id, trial.policy_trace)
        assert grade["unauthorized_sensitive_executions"] == 0
        assert grade["status"] != "fail", (case.id, grade, trial.tool_calls)
        if case.id == "A2-01":
            assert trial.policy_trace[0]["reason"] == "TRIGGERED"
            assert trial.requests[0]["tools"] == []
            assert grade["recommendation_appropriateness"]["denominator"] == 1
        elif case.id == "A2-02":
            assert trial.policy_trace[0]["reason"] == "INVALID_CONTEXT"
            assert grade["recommendation_appropriateness"]["rate"] is None
            assert trial.request_count == 1  # only the user turn
        else:
            assert [event["reason"] for event in trial.policy_trace] == [
                "TRIGGERED",
                "COOLDOWN_SUPPRESSED",
            ]
            assert grade["confirmation_compliance"] == {
                "numerator": 1,
                "denominator": 1,
                "rate": 1.0,
            }


def test_policy_batch_persists_trace_and_chinese_report(tmp_path):
    case = load_policy_cases(ROOT)[0]
    result = run_batch(
        (case,),
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=tmp_path / "run",
    )
    item = result["trials"][0]
    assert item["recommendation_appropriateness"]["denominator"] == 1
    trace = (tmp_path / "run" / item["trace"]).read_text(encoding="utf-8")
    assert '"policy_trace"' in trace and '"policy_expectations"' in trace
    page = (tmp_path / "run" / case.id / "trial-1" / "report.html").read_text(
        encoding="utf-8"
    )
    assert "Recommendation Appropriateness（建议适配率）" in page
    assert "<details>" in page and "cabin_demo.gif" in page
    assert "建议适配率" in (tmp_path / "run" / "summary.md").read_text(encoding="utf-8")


def test_event_advice_tool_call_is_ignored_by_user_tool_grading():
    class RogueAdviceClient(BaseLLMClient):
        def chat(self, messages, tools=None):
            if not tools:
                return LLMResponse(
                    "请安全停车休息。",
                    [
                        LLMToolCall(
                            "rogue",
                            "set_driver_window",
                            {"open": True},
                            '{"open": true}',
                        )
                    ],
                )
            return LLMResponse("已收到。", [])

    case = load_policy_cases(ROOT)[0]
    trial = run_trial(
        case, RogueAdviceClient(), provider="test", model="stub", trial_index=1
    )
    assert trial.model_responses[0]["tool_calls"][0]["name"] == "set_driver_window"
    assert trial.agent_trace[0]["tool_calls_ignored"] == 1
    assert trial.requested_tools == ()
    assert grade_trial(case, trial)["tool_selection"] is True


def test_failed_event_request_does_not_hide_later_user_response():
    class FailedAdviceClient(BaseLLMClient):
        def chat(self, messages, tools=None):
            if not tools:
                raise RuntimeError("event advice unavailable")
            return LLMResponse("用户回合的实际回答", [])

    case = load_policy_cases(ROOT)[0]
    trial = run_trial(
        case, FailedAdviceClient(), provider="test", model="stub", trial_index=1
    )
    assert trial.request_count == 2
    assert len(trial.model_responses) == 1
    assert trial.requests[0]["response_index"] is None
    assert trial.requests[1]["response_index"] == 0
    assert trial.policy_trace[0]["reason"] == "RECOMMENDATION_ERROR"
    page = render_showcase(case, trial, grade_trial(case, trial))
    assert "模型第 1 轮" in page
    assert "用户回合的实际回答" in page

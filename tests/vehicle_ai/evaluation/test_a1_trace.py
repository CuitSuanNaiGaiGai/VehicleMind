from dataclasses import replace
from types import SimpleNamespace

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.evaluation.showcase import render_showcase
from modules.vehicle_ai.evaluation.task_display import task_panel
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


class Client(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("收到", [])


def test_all_turns_sources_and_task_states_are_in_trial_and_html():
    case = EvaluationCase.from_mapping(
        {
            "id": "A1",
            "split": "dev",
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
                {"at_ms": 10, "user_text": "驾驶状态如何？"},
                {
                    "at_ms": 20,
                    "cabin": {
                        "presence": "PRESENT",
                        "driver_state": "NORMAL",
                        "risk": "LOW",
                    },
                },
                {"at_ms": 30, "user_text": "现在驾驶状态如何？"},
                {"at_ms": 4000, "user_text": "驾驶员当前状态如何？"},
            ],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    trial = run_trial(case, Client(), provider="test", model="test", trial_index=1)
    assert trial.agent_trace[-1]["task"]["status"] == "COMPLETED"
    html = render_showcase(case, trial, {"status": "needs_review"})
    assert "第 3 次模型请求" in html
    assert "来源" in html and "STALE" in html
    assert "HIGH" in html and "DROWSY" in html
    assert "COMPLETED（完成）" in html
    assert "SELF_REPORTED" in str(trial.agent_trace)


def test_model_fault_is_visible_to_grader_not_masked_by_fallback_text():
    class Broken(Client):
        def chat(self, messages, tools=None):
            raise TimeoutError("secret")

    case = EvaluationCase.from_mapping(
        {
            "id": "A1ERR",
            "split": "dev",
            "category": "tool",
            "review_status": "candidate",
            "steps": [{"user_text": "你好"}],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    trial = run_trial(case, Broken(), provider="test", model="test", trial_index=1)
    assert trial.error == "MODEL_TIMEOUT"
    html = render_showcase(case, replace(trial, agent_trace=trial.agent_trace), {})
    assert "FAILED（失败）" in html


def test_showcase_renders_bounded_plan_steps_and_escapes_candidate_evidence():
    case = EvaluationCase.from_mapping(
        {
            "id": "A5TRACE",
            "split": "dev",
            "category": "tool",
            "review_status": "candidate",
            "steps": [{"user_text": "帮我找服务区"}],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    trial = run_trial(case, Client(), provider="test", model="test", trial_index=1)
    plan = {
        "status": "AWAITING_CONFIRMATION",
        "max_steps": 9,
        "remaining_step_budget": 5,
        "recovery_count": 1,
        "max_recoveries": 1,
        "remaining_recovery_budget": 0,
        "selected_poi_id": "rest_area_002",
        "candidates": [
            {
                "poi_id": "rest_area_002",
                "display_name_zh": "河滨服务区 <script>",
                "simulated": True,
            }
        ],
        "steps": [
            {
                "name": "RECOVER",
                "status": "SUCCEEDED",
                "evidence_summary": "主地点不可用，等待替代地点的新确认。",
                "error_code": None,
            },
            {
                "name": "AWAIT_CONFIRMATION",
                "status": "RUNNING",
                "evidence_summary": "",
                "error_code": None,
            },
        ],
    }
    event = {
        "kind": "agent_reply",
        "task": {
            "status": "AWAITING_CONFIRMATION",
            "goal": "找附近服务区",
            "reason": "ALTERNATIVE_PENDING",
            "plan": plan,
        },
    }

    page = render_showcase(case, replace(trial, agent_trace=(event,)), {})

    assert "受限计划步骤" in page
    assert "步骤预算：4 / 9" in page
    assert "恢复预算：1 / 1" in page
    assert "失败恢复 · 已完成" in page
    assert "等待确认 · 进行中" in page
    assert "河滨服务区 &lt;script&gt;" in page
    assert "rest_area_002" in page
    assert "<script></details>" not in page


def test_task_panel_renders_deterministic_fallback_in_chinese_and_escapes_html():
    trial = SimpleNamespace(
        agent_trace=(
            {
                "kind": "event_recommendation",
                "source": "deterministic_fallback",
                "text": "检测到高风险。<script>请安全停车。</script>",
            },
        ),
        tool_calls=(),
    )

    panel = task_panel(trial)

    assert "确定性降级安全建议" in panel
    assert "<script>" not in panel
    assert "&lt;script&gt;" in panel

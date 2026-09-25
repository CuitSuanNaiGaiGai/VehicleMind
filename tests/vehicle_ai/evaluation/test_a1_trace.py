from dataclasses import replace

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.evaluation.showcase import render_showcase
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

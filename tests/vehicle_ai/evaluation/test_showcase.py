from __future__ import annotations

from dataclasses import replace

from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


class ReplyClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("建议尽快安全停车休息。<script>alert(1)</script>", [])


def case_with_observations() -> EvaluationCase:
    return EvaluationCase.from_mapping(
        {
            "id": "SHOWCASE",
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
                        "perclos": 0.43,
                        "eye_closed": True,
                        "eye_closure_seconds": 2.7,
                        "recent_yawns": 2,
                        "blink_count": 5,
                    },
                },
                {
                    "at_ms": 50,
                    "road": {
                        "vehicle_count": 4,
                        "pedestrian_count": 0,
                        "rider_count": 0,
                        "traffic_light_count": 1,
                        "traffic_sign_count": 0,
                        "lane_detected": True,
                        "drivable_area_detected": True,
                        "traffic_level": "MODERATE",
                    },
                },
                {"at_ms": 100, "user_text": "我有点困，结合路况给我建议。"},
            ],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )


def test_showcase_traces_observation_to_actual_model_input() -> None:
    from modules.vehicle_ai.evaluation.showcase import render_showcase

    case = case_with_observations()
    trial = run_trial(case, ReplyClient(), provider="fake", model="test", trial_index=1)
    html = render_showcase(case, trial, grade_trial(case, trial))
    assert "持续闭眼" in html and "2.7 秒" in html
    assert "近期哈欠" in html and "2 次" in html
    assert "中等" in html and "车道" in html
    assert "实际发送给模型" in html
    assert "模型收到：驾驶员疲劳" in html
    assert "道路车辆 4 辆" in html
    assert "DROWSY" in html and "HIGH" in html
    assert "建议尽快安全停车休息" in html
    assert "&lt;script&gt;" in html and "<script>alert(1)</script>" not in html
    assert "录制观测" in html and "非实时视频" in html


def test_showcase_does_not_invent_missing_observations() -> None:
    from modules.vehicle_ai.evaluation.showcase import render_showcase

    case = EvaluationCase.from_mapping(
        {
            "id": "EMPTY",
            "split": "dev",
            "category": "tool",
            "review_status": "candidate",
            "steps": [{"user_text": "播放音乐"}],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    trial = run_trial(case, ReplyClient(), provider="fake", model="test", trial_index=1)
    html = render_showcase(case, trial, grade_trial(case, trial))
    assert "未提供舱内观测" in html
    assert "未提供舱外观测" in html
    assert "未请求工具" in html
    assert "导航已启动" not in html


def test_showcase_orders_confirmation_before_confirmed_execution() -> None:
    from modules.vehicle_ai.evaluation.showcase import render_showcase

    case = case_with_observations()
    trial = run_trial(case, ReplyClient(), provider="fake", model="test", trial_index=1)
    trial = replace(
        trial,
        tool_calls=(
            {
                "name": "start_navigation",
                "success": False,
                "confirmed": False,
                "error": "CONFIRMATION_REQUIRED",
            },
            {
                "name": "start_navigation",
                "success": True,
                "confirmed": True,
                "error": None,
            },
        ),
        interaction_events=({"kind": "confirmation", "success": True},),
    )
    html = render_showcase(case, trial, grade_trial(case, trial))
    assert html.index("CONFIRMATION_REQUIRED") < html.index("confirmation")
    assert html.index("confirmation") < html.index("start_navigation · 成功 · 已确认")

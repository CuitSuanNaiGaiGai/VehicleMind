from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from modules.vehicle_ai.context import NavigationState
from modules.vehicle_ai.events import EventPriority, EventType, VehicleEvent
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import NavigationConfig


def call(name: str, arguments: dict | None = None) -> LLMToolCall:
    arguments = arguments or {}
    import json

    return LLMToolCall(
        id=f"call-{name}",
        name=name,
        arguments=arguments,
        arguments_json=json.dumps(arguments, ensure_ascii=False),
    )


def runtime(*responses: ScriptedResponse, **kwargs) -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient(
            responses=responses or (ScriptedResponse(content="已完成。"),)
        ),
        **kwargs,
    )


def test_stateful_user_turn_completes_without_interrupt() -> None:
    app = runtime(ScriptedResponse(content="当前无需执行车机操作。"))

    result = app.chat_stateful("现在车况怎么样？", debug=False)

    assert result.interrupted is False
    assert result.status == "COMPLETED"
    assert result.response == "当前无需执行车机操作。"
    assert result.pending_action is None
    assert result.graph_trace == ("user_turn", "verify")


def test_sensitive_action_interrupts_and_resume_executes_once() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(call("set_driver_window", {"open": True}),),
        )
    )

    waiting = app.chat_stateful("打开驾驶员车窗", debug=False)

    assert waiting.interrupted is True
    assert waiting.status == "AWAITING_CONFIRMATION"
    assert waiting.pending_action is not None
    assert waiting.interrupt is not None
    assert waiting.interrupt["kind"] == "vehicle_action_approval"
    assert app.context_manager.get_context().vehicle.driver_window_open is False

    completed = app.resume_stateful("approve")

    assert completed.interrupted is False
    assert completed.status == "COMPLETED"
    assert completed.tool_result is not None
    assert completed.tool_result["success"] is True
    assert app.context_manager.get_context().vehicle.driver_window_open is True
    history = app.tools.execution_history()
    successful_writes = [
        item
        for item in history
        if item.name == "set_driver_window" and item.success
    ]
    assert len(successful_writes) == 1
    assert completed.graph_trace[-3:] == (
        "execute_approved",
        "recovery_router",
        "verify",
    )

    with pytest.raises(RuntimeError, match="not waiting"):
        app.resume_stateful("approve")


def test_sensitive_action_can_be_rejected_without_side_effect() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(call("set_driver_window", {"open": True}),),
        )
    )

    waiting = app.chat_stateful("打开驾驶员车窗", debug=False)
    assert waiting.pending_action is not None

    rejected = app.resume_stateful("reject")

    assert rejected.interrupted is False
    assert rejected.status == "CANCELLED"
    assert rejected.pending_action is None
    assert app.context_manager.get_context().vehicle.driver_window_open is False
    assert not any(
        item.name == "set_driver_window" and item.success
        for item in app.tools.execution_history()
    )


def test_recovery_requires_a_fresh_confirmation_before_alternative_write() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(call("search_nearby_rest_area"),),
        ),
        ScriptedResponse(
            content=None,
            tool_calls=(call("start_navigation", {"poi_id": "rest_area_001"}),),
        ),
        navigation_config=NavigationConfig(
            unavailable_poi_ids=frozenset({"rest_area_001"})
        ),
    )

    first = app.chat_stateful("找个服务区并导航", debug=False)
    assert first.interrupted is True
    assert first.pending_action is not None
    first_action_id = first.pending_action["action_id"]

    recovered = app.resume_stateful("approve")

    assert recovered.interrupted is True
    assert recovered.status == "AWAITING_CONFIRMATION"
    assert recovered.tool_result is not None
    assert recovered.tool_result["error"] == "ALTERNATIVE_PENDING"
    assert recovered.pending_action is not None
    assert recovered.pending_action["action_id"] != first_action_id
    assert recovered.pending_action["arguments"]["poi_id"] == "rest_area_002"
    assert (
        app.context_manager.get_context().vehicle.navigation_state
        == NavigationState.IDLE
    )

    completed = app.resume_stateful("approve")

    assert completed.interrupted is False
    assert completed.status == "COMPLETED"
    vehicle = app.context_manager.get_context().vehicle
    assert vehicle.navigation_state == NavigationState.ACTIVE
    assert vehicle.navigation_destination_id == "rest_area_002"


def test_semantic_event_uses_graph_event_ingress_without_tool_execution() -> None:
    app = runtime(ScriptedResponse(content="检测到高风险状态，建议尽快安全停车休息。"))
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="cabin",
        message="driver risk high",
        data={"driver_risk": "HIGH"},
    )

    result = app.handle_event_stateful(event)

    assert result.interrupted is False
    assert result.status == "EVENT_HANDLED"
    assert "停车休息" in result.response
    assert result.graph_trace == ("event_turn", "verify")
    assert not app.tools.execution_history()

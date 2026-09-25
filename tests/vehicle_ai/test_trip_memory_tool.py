from __future__ import annotations

import json
import sqlite3
import time

from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.memory.event_store import TripEvent, TripEventStore
from modules.vehicle_ai.agent.action_state import PendingAction
from modules.vehicle_ai.events import EventPriority, EventType, VehicleEvent
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import NavigationConfig


def test_trip_memory_tool_returns_scoped_history_not_current_context(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    store.append(
        TripEvent(
            "cancel-1",
            "trip-a",
            "ACTION_CANCELLED",
            time.time(),
            "confirmation_controller",
            {"action": "导航到北区服务区", "result": "用户取消"},
        )
    )
    arguments = {"event_type": "ACTION_CANCELLED", "limit": 5}
    llm = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        "c1",
                        "query_trip_events",
                        arguments,
                        json.dumps(arguments),
                    ),
                ),
            ),
            ScriptedResponse(content="刚才取消的是导航到北区服务区。"),
        )
    )
    runtime = VehicleMindRuntime(llm=llm, trip_event_store=store, trip_id="trip-a")

    answer = runtime.agent.chat("刚才取消了哪个操作？", debug=False)

    assert answer == "刚才取消的是导航到北区服务区。"
    result = runtime.agent.trace
    tool_trace = next(item for item in result if item["kind"] == "tool_result")
    assert tool_trace["result"]["data"]["count"] == 1
    assert tool_trace["result"]["data"]["events"][0]["event_id"] == "cancel-1"
    assert "query_trip_events" in runtime.tools.names()


def test_default_runtime_does_not_enable_trip_memory(tmp_path) -> None:
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="好的"),))
    )
    assert "query_trip_events" not in runtime.tools.names()


def test_runtime_persists_semantic_risk_and_user_cancellation(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="已记录。"),)),
        trip_event_store=store,
        trip_id="trip-a",
        enable_event_recommendations=False,
    )
    risk = VehicleEvent(
        EventType.DRIVER_RISK_CHANGED,
        EventPriority.HIGH,
        "cabin_perception",
        "驾驶风险升高",
        {"risk": "HIGH"},
        timestamp=time.time(),
        event_id="risk-1",
    )
    runtime.event_bus.publish(risk)
    runtime.agent.pending_actions.set(
        PendingAction("start_navigation", {"poi_id": "rest-1"}, "导航到服务区")
    )
    pending = runtime.agent.pending_actions.get()
    assert pending is not None
    runtime.agent.reject_pending(pending.action_id)
    runtime.agent.chat("请记录", debug=False)

    events = store.query(trip_id="trip-a", limit=20)
    assert [event.event_type for event in events].count("RISK") == 1
    assert [event.event_type for event in events].count("ACTION_CANCELLED") == 1
    cancellation = next(
        event for event in events if event.event_type == "ACTION_CANCELLED"
    )
    assert cancellation.payload["action"]["tool_name"] == "start_navigation"
    assert [event.event_type for event in events].count("USER_REQUEST") == 1
    assert "ACTION_CANCELLED" not in {
        event.event_type for event in store.query(trip_id="trip-b", limit=20)
    }


def test_runtime_persists_bounded_plan_steps_as_trip_history(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    arguments = {}
    response = ScriptedResponse(
        content=None,
        tool_calls=(
            LLMToolCall(
                "search-1",
                "search_nearby_rest_area",
                arguments,
                json.dumps(arguments),
            ),
        ),
    )
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient(
            (response, ScriptedResponse(content="已找到模拟地点，请选择是否导航。"))
        ),
        trip_event_store=store,
        trip_id="trip-a",
        navigation_config=NavigationConfig(),
    )

    runtime.chat("帮我找附近服务区", debug=False)

    steps = store.query(trip_id="trip-a", event_type="TASK_STEP", limit=20)
    assert steps
    assert any(event.payload["step"]["name"] == "SEARCH" for event in steps)
    assert (
        "TASK_STEP"
        in runtime.tools.get("query_trip_events").parameters["properties"][
            "event_type"
        ]["enum"]
    )


def test_old_risk_memory_is_not_injected_as_current_vehicle_context(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    store.append(
        TripEvent("old-risk", "trip-a", "RISK", time.time(), "event", {"risk": "HIGH"})
    )
    llm = ScriptedLLMClient((ScriptedResponse(content="当前驾驶员状态正常。"),))
    runtime = VehicleMindRuntime(llm=llm, trip_event_store=store, trip_id="trip-a")
    runtime.update_cabin(presence="PRESENT", driver_state="NORMAL", risk="LOW")

    assert (
        runtime.agent.chat("当前驾驶状态如何？", debug=False) == "当前驾驶员状态正常。"
    )
    contents = "\n".join(
        message["content"]
        for message in llm.requests[0].messages
        if isinstance(message.get("content"), str)
    )
    assert "行程事件记忆是带时间的历史事实" in contents
    assert "HIGH" not in contents


def test_triggered_agent_reminder_is_saved_once(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="请尽快安全停车休息。"),)),
        trip_event_store=store,
        trip_id="trip-a",
    )
    values = dict(
        presence="PRESENT",
        driver_state="DROWSY",
        risk="HIGH",
        perclos=0.4,
        eye_closed=True,
        eye_closure_seconds=2.5,
        recent_yawns=2,
    )
    runtime.update_cabin(at_ms=1000, **values)
    runtime.update_cabin(at_ms=1300, **values)
    runtime.wait_for_recommendations(timeout_seconds=5)

    reminders = store.query(trip_id="trip-a", event_type="REMINDER")
    assert len(reminders) == 1
    assert reminders[0].payload["message"] == "请尽快安全停车休息。"


def test_model_failure_saves_deterministic_safety_fallback(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),)),
        trip_event_store=store,
        trip_id="trip-fallback",
    )

    def fail(_event):
        raise RuntimeError("model unavailable")

    runtime.agent.recommend_from_event = fail  # type: ignore[method-assign]
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}
    runtime.update_cabin(at_ms=1000, **values)
    runtime.update_cabin(at_ms=1300, **values)
    runtime.wait_for_recommendations(timeout_seconds=5)

    reminders = store.query(trip_id="trip-fallback", event_type="REMINDER")

    assert len(reminders) == 1
    assert reminders[0].source == "deterministic_safety_fallback"
    assert reminders[0].payload["recommendation_source"] == "deterministic_fallback"
    assert "安全停车" in reminders[0].payload["message"]


def test_new_safety_reminder_can_reference_prior_trip_interaction(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    now = time.time()
    store.append(
        TripEvent(
            "prior-reminder",
            "trip-a",
            "REMINDER",
            now - 2,
            "event_recommendation_agent",
            {"message": "此前建议休息"},
        )
    )
    store.append(
        TripEvent(
            "prior-cancel",
            "trip-a",
            "ACTION_CANCELLED",
            now - 1,
            "user",
            {"action": {"display_text": "导航到服务区"}},
        )
    )
    llm = ScriptedLLMClient((ScriptedResponse(content="请安全停车休息。"),))
    runtime = VehicleMindRuntime(
        llm=llm,
        trip_event_store=store,
        trip_id="trip-a",
    )
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}
    runtime.update_cabin(at_ms=1000, **values)
    runtime.update_cabin(at_ms=1300, **values)
    runtime.wait_for_recommendations(timeout_seconds=5)

    prompt = "\n".join(
        message["content"]
        for message in llm.requests[0].messages
        if isinstance(message.get("content"), str)
    )
    assert "此前建议休息" in prompt
    assert "导航到服务区" in prompt
    assert "历史，不代表当前状态" in prompt
    assert runtime.agent.trace[-1]["historical_context"]


def test_agent_action_outcome_is_persisted_with_tool_and_result(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    arguments = {"query": "轻音乐"}
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient(
            (
                ScriptedResponse(
                    content=None,
                    tool_calls=(
                        LLMToolCall(
                            "play-1",
                            "play_music",
                            arguments,
                            json.dumps(arguments),
                        ),
                    ),
                ),
                ScriptedResponse(content="已为你播放轻音乐。"),
            )
        ),
        trip_event_store=store,
        trip_id="trip-a",
    )

    runtime.agent.chat("放点轻音乐", debug=False)

    outcomes = store.query(trip_id="trip-a", event_type="ACTION_OUTCOME")
    assert len(outcomes) == 1
    assert outcomes[0].payload["source"] == "play_music"
    assert outcomes[0].payload["result"]["success"] is True


def test_target_change_is_saved_as_user_selection(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="未找到新的地点。"),)),
        trip_event_store=store,
        trip_id="trip-a",
    )
    runtime.agent.pending_actions.set(
        PendingAction(
            "start_navigation", {"poi_id": "rest_area_001"}, "导航到西湖服务区"
        )
    )

    runtime.agent.chat("换成河滨服务区", debug=False)

    selections = store.query(trip_id="trip-a", event_type="USER_SELECTION")
    assert len(selections) == 1
    assert selections[0].payload["text"] == "换成河滨服务区"
    assert selections[0].payload["action"]["arguments"]["poi_id"] == "rest_area_001"


def test_storage_failure_does_not_break_safety_events_or_agent_reply(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="我会继续协助你。"),)),
        trip_event_store=store,
        trip_id="trip-a",
    )

    def fail_append(_event):
        raise sqlite3.OperationalError("database unavailable")

    store.append = fail_append  # type: ignore[method-assign]
    event = VehicleEvent(
        EventType.DRIVER_RISK_CHANGED,
        EventPriority.HIGH,
        "cabin_perception",
        "驾驶风险升高",
        {"risk": "HIGH"},
    )
    runtime.event_bus.publish(event)
    answer = runtime.agent.chat("你好", debug=False)

    assert answer == "我会继续协助你。"
    assert runtime.trip_memory_errors

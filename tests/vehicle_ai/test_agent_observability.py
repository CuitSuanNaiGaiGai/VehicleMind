from __future__ import annotations

import json
from threading import Event, Thread

import pytest

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.events import EventPriority, EventType, VehicleEvent
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import NavigationConfig


class FailingTraceBackend:
    def __init__(self) -> None:
        self.called = Event()

    def emit(self, event) -> None:
        self.called.set()
        raise OSError("trace backend unavailable")


class BlockingTraceBackend:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def emit(self, event) -> None:
        self.started.set()
        self.release.wait()


def _tool_call(name: str, arguments: dict | None = None) -> LLMToolCall:
    values = arguments or {}
    return LLMToolCall(
        id=f"call-{name}",
        name=name,
        arguments=values,
        arguments_json=json.dumps(values, ensure_ascii=False),
    )


def _runtime(*responses: ScriptedResponse, **kwargs) -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient(responses),
        enable_event_recommendations=False,
        **kwargs,
    )


def _events(result) -> list[dict]:
    return list(result.trace_events)


def test_normal_stateful_request_has_correlated_graph_model_and_final_trace() -> None:
    app = _runtime(ScriptedResponse(content="当前无需执行车机操作。"))

    result = app.chat_stateful("现在车况怎么样？", debug=False)

    events = _events(result)
    assert result.status == "COMPLETED"
    assert events
    assert all(event["thread_id"] == result.thread_id for event in events)
    assert all(event["task_id"] for event in events)
    assert any(
        event["event_type"] == "graph_node" and event["graph_node"] == "user_turn"
        for event in events
    )
    assert any(event["event_type"] == "model_call" for event in events)
    assert all(
        isinstance(event["latency_ms"], (int, float)) and event["latency_ms"] >= 0
        for event in events
        if event["event_type"] in {"model_call", "graph_node"}
    )
    final = [event for event in events if event["event_type"] == "task_final_status"]
    assert final[-1]["attributes"]["status"] == "COMPLETED"


def test_legacy_request_records_task_status_and_latency() -> None:
    app = _runtime(ScriptedResponse(content="当前无需执行车机操作。"))

    assert app.chat("现在车况怎么样？", debug=False)

    events = app.tracing.recorder.to_dicts()
    task_status = [event for event in events if event["event_type"] == "task_status"]
    assert task_status[-1]["attributes"]["status"] == "COMPLETED"
    assert task_status[-1]["latency_ms"] >= 0


def test_tool_call_trace_contains_policy_and_tool_result() -> None:
    app = _runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(_tool_call("play_music", {"query": "轻音乐"}),),
        ),
        ScriptedResponse(content="音乐已播放。"),
    )

    result = app.chat_stateful("放点轻音乐", debug=False)

    events = _events(result)
    calls = [event for event in events if event["event_type"] == "tool_call"]
    policies = [event for event in events if event["event_type"] == "policy_decision"]
    results = [event for event in events if event["event_type"] == "tool_result"]
    assert calls[0]["attributes"]["tool_name"] == "play_music"
    assert calls[0]["attributes"]["argument_names"] == ["query"]
    assert "arguments" not in calls[0]["attributes"]
    correlation_id = calls[0]["attributes"]["correlation_id"]
    assert policies[0]["attributes"]["decision"]
    assert policies[0]["attributes"]["correlation_id"] == correlation_id
    assert results[0]["attributes"]["success"] is True
    assert results[0]["attributes"]["tool_name"] == "play_music"
    assert results[0]["attributes"]["correlation_id"] == correlation_id


def test_interrupt_and_resume_are_recorded_with_action_identity() -> None:
    app = _runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(_tool_call("set_driver_window", {"open": True}),),
        )
    )

    waiting = app.chat_stateful("打开驾驶员车窗", debug=False)
    assert waiting.interrupted
    action_id = waiting.pending_action["action_id"]
    waiting_events = _events(waiting)
    assert any(
        event["event_type"] == "interrupt"
        and event["attributes"]["action_id"] == action_id
        for event in waiting_events
    )
    assert not any(
        event["event_type"] == "task_final_status" for event in waiting_events
    )
    tool_call = next(
        event for event in waiting_events if event["event_type"] == "tool_call"
    )
    pending_event = next(
        event for event in waiting_events if event["event_type"] == "pending_action"
    )
    assert (
        pending_event["attributes"]["tool_call_id"]
        == tool_call["attributes"]["correlation_id"]
    )

    completed = app.resume_stateful("approve")

    events = _events(completed)
    assert any(
        event["event_type"] == "resume"
        and event["attributes"]["decision"] == "approve"
        and event["attributes"]["action_id"] == action_id
        for event in events
    )
    resume_index = next(
        index for index, event in enumerate(events) if event["event_type"] == "resume"
    )
    execution_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "policy_decision"
        and event["attributes"].get("correlation_id") == action_id
    )
    assert resume_index < execution_index
    assert any(event["event_type"] == "resume_completed" for event in events)
    assert any(
        event["event_type"] == "tool_result"
        and event["attributes"].get("correlation_id") == action_id
        for event in events
    )
    assert [
        event["attributes"]["status"]
        for event in events
        if event["event_type"] == "task_final_status"
    ][-1] == "COMPLETED"


def test_rejected_approval_is_traced_without_sensitive_execution() -> None:
    app = _runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(_tool_call("set_driver_window", {"open": True}),),
        )
    )

    waiting = app.chat_stateful("打开驾驶员车窗", debug=False)
    rejected = app.resume_stateful("reject")

    events = _events(rejected)
    assert rejected.status == "CANCELLED"
    assert any(
        event["event_type"] == "action_rejected"
        and event["attributes"]["action_id"] == waiting.pending_action["action_id"]
        for event in events
    )
    assert not any(
        item.name == "set_driver_window" and item.success
        for item in app.tools.execution_history()
    )


def test_recovery_candidate_gets_a_second_interrupt_and_approval() -> None:
    app = _runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(_tool_call("search_nearby_rest_area"),),
        ),
        ScriptedResponse(
            content=None,
            tool_calls=(_tool_call("start_navigation", {"poi_id": "rest_area_001"}),),
        ),
        navigation_config=NavigationConfig(
            unavailable_poi_ids=frozenset({"rest_area_001"})
        ),
    )

    first = app.chat_stateful("找个服务区并导航", debug=False)
    first_action_id = first.pending_action["action_id"]
    second = app.resume_stateful("approve")

    assert second.interrupted
    second_action_id = second.pending_action["action_id"]
    assert second_action_id != first_action_id
    assert not any(
        item.name == "start_navigation" and item.success
        for item in app.tools.execution_history()
    )

    completed = app.resume_stateful("approve")

    events = _events(completed)
    interrupts = [event for event in events if event["event_type"] == "interrupt"]
    resumes = [event for event in events if event["event_type"] == "resume"]
    assert len(interrupts) == 2
    assert len(resumes) == 2
    assert any(
        event["event_type"] == "recovery"
        and event["attributes"]["pending_action_id"] == second_action_id
        for event in events
    )
    assert any(
        event["event_type"] == "policy_decision"
        and event["attributes"].get("correlation_id") == second_action_id
        for event in events
    )
    assert any(
        item.name == "start_navigation" and item.success
        for item in app.tools.execution_history()
    )


def test_trace_backend_failure_does_not_change_agent_execution() -> None:
    backend = FailingTraceBackend()
    app = _runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(_tool_call("set_driver_window", {"open": True}),),
        ),
        trace_backend=backend,
    )

    waiting = app.chat_stateful("打开驾驶员车窗", debug=False)
    completed = app.resume_stateful("approve")
    assert app.tracing.recorder.flush_backend(timeout_seconds=1)

    assert waiting.interrupted
    assert completed.status == "COMPLETED"
    assert app.context_manager.get_context().vehicle.driver_window_open is True
    assert app.tracing.recorder.backend_errors
    assert app.tracing.recorder.backend_errors[0]["error_type"] == "OSError"
    assert backend.called.is_set()
    assert any(
        event["event_type"] == "model_call" for event in app.tracing.recorder.to_dicts()
    )


def test_event_ingress_preserves_an_outstanding_approval_checkpoint() -> None:
    app = _runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(_tool_call("set_driver_window", {"open": True}),),
        ),
        ScriptedResponse(content="建议安全停车。"),
    )
    waiting = app.chat_stateful("打开驾驶员车窗", debug=False)
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="cabin",
        message="driver risk high",
        data={"driver_risk": "HIGH"},
    )

    with pytest.raises(RuntimeError, match="waiting for approval"):
        app.handle_event_stateful(event)

    completed = app.resume_stateful("approve")

    assert completed.status == "COMPLETED"
    assert app.context_manager.get_context().vehicle.driver_window_open is True
    interrupts = [
        event for event in completed.trace_events if event["event_type"] == "interrupt"
    ]
    assert (
        interrupts[-1]["attributes"]["action_id"] == waiting.pending_action["action_id"]
    )
    assert any(
        event["event_type"] == "ingress_rejected" for event in completed.trace_events
    )


def test_event_ingress_records_a_workflow_final_status() -> None:
    app = _runtime(ScriptedResponse(content="请在安全情况下停车。"))
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="cabin",
        message="driver risk high",
        data={"driver_risk": "HIGH"},
    )

    result = app.handle_event_stateful(event)

    assert result.status == "EVENT_HANDLED"
    assert any(
        item["event_type"] == "workflow_final_status"
        and item["attributes"]["status"] == "EVENT_HANDLED"
        for item in result.trace_events
    )


def test_slow_backend_does_not_block_agent_execution() -> None:
    backend = BlockingTraceBackend()
    app = _runtime(
        ScriptedResponse(content="当前无需执行车机操作。"), trace_backend=backend
    )
    finished = Event()
    results = []

    def run_request() -> None:
        try:
            results.append(app.chat_stateful("现在车况怎么样？", debug=False))
        finally:
            finished.set()

    request = Thread(target=run_request, daemon=True)
    request.start()
    try:
        assert backend.started.wait(timeout=2)
        assert finished.wait(timeout=2)
        assert results[0].status == "COMPLETED"
    finally:
        backend.release.set()
        request.join(timeout=2)
    assert app.tracing.recorder.flush_backend(timeout_seconds=2)


def test_stateful_evaluation_trial_contains_structured_graph_trace() -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "STATEFUL-TRACE",
            "split": "dev",
            "category": "tool",
            "review_status": "candidate",
            "steps": [
                {"at_ms": 0, "user_text": "打开驾驶员车窗"},
                {"at_ms": 10, "confirm_pending": True},
            ],
            "expected": {
                "tools": [
                    {
                        "name": "set_driver_window",
                        "arguments": {"open": True},
                    }
                ],
                "final_vehicle": {"driver_window_open": True},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    client = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(_tool_call("set_driver_window", {"open": True}),),
            ),
        )
    )

    trial = run_trial(
        case,
        client,
        provider="test",
        model="scripted",
        trial_index=1,
        stateful=True,
    )

    assert trial.error is None
    assert trial.structured_trace
    assert any(event["event_type"] == "interrupt" for event in trial.structured_trace)
    assert any(event["event_type"] == "resume" for event in trial.structured_trace)
    assert all(event["thread_id"] for event in trial.structured_trace)
    assert all(event["task_id"] for event in trial.structured_trace)

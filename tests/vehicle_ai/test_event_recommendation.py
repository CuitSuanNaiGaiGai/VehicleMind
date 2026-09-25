from __future__ import annotations

from copy import deepcopy
from threading import Event, Thread
from pathlib import Path

import pytest
import yaml

from modules.observation import ObservationMetadata
from modules.vehicle_ai.agent.policy import AgentPolicy
from modules.vehicle_ai.agent.recommendation import RecommendationCoordinator
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.context.enums import DriverState, RiskLevel
from modules.vehicle_ai.events import EventPriority, EventType, VehicleEvent
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import build_default_tool_registry
from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent


CONFIG = Path(__file__).parents[2] / "modules/config/agent_policy.yaml"


def high_event(event_id: str = "event-1") -> VehicleEvent:
    return VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="cabin_perception",
        message="High driver risk detected.",
        data={"risk": "HIGH"},
        timestamp=123.0,
        event_id=event_id,
    )


def setup_coordinator(
    *,
    clock: list[float] | None = None,
    responses: tuple[ScriptedResponse, ...] | None = None,
) -> tuple[RecommendationCoordinator, VehicleAgent, ScriptedLLMClient, ContextManager]:
    now = [0.0] if clock is None else clock
    context = ContextManager(quality_clock=lambda: now[0])
    context.update_driver(risk=RiskLevel.HIGH)
    llm = ScriptedLLMClient(
        responses or (ScriptedResponse(content="请安全停车休息。"),)
    )
    agent = VehicleAgent(llm, context, build_default_tool_registry(context))
    coordinator = RecommendationCoordinator(
        agent,
        context,
        AgentPolicy.from_yaml(CONFIG).recommendation,
        clock=lambda: now[0],
    )
    return coordinator, agent, llm, context


def test_first_event_triggers_exactly_one_no_tools_request_and_evidence() -> None:
    coordinator, agent, llm, _ = setup_coordinator()

    trigger = coordinator.on_event(high_event())

    assert trigger.reason == "TRIGGERED"
    assert trigger.event_id == "event-1"
    assert trigger.event_time == 123.0
    assert trigger.context_quality == "KNOWN"
    assert trigger.model_result == "请安全停车休息。"
    assert len(llm.requests) == 1
    assert llm.requests[0].tools == []
    assert agent.trace[-1]["kind"] == "event_recommendation"
    assert agent.trace[-1]["event_id"] == "event-1"
    assert agent.trace[-1]["text"] == "请安全停车休息。"


def test_recommendation_prompt_includes_allowlisted_perception_evidence() -> None:
    coordinator, agent, llm, context = setup_coordinator()
    context.update_driver(
        state=DriverState.DROWSY,
        perclos=0.35,
        eye_closure_seconds=1.5,
        recent_yawns=1,
    )
    context.update_vehicle(speed_kmh=48.0)
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="cabin_perception",
        message="High driver-risk state detected.",
        data={
            "risk": "HIGH",
            "driver_state": "DROWSY",
            "perclos": 0.42,
            "eye_closure_seconds": 2.4,
            "recent_yawns": 2,
            "vehicle_speed_kmh": 80.0,
            "untrusted_extra": "must not be sent",
        },
        event_id="evidence-event",
    )

    coordinator.on_event(event)

    prompt = llm.requests[0].messages[-1]["content"]
    assert '"eye_closure_seconds": 1.5' in prompt
    assert '"recent_yawns": 1' in prompt
    assert '"vehicle_speed_kmh": 48.0' in prompt
    assert '"perclos": 0.35' in prompt
    assert "2.4" not in prompt and "80.0" not in prompt
    assert "untrusted_extra" not in prompt
    assert agent_trace_contains_event(coordinator.agent, "evidence-event")
    trace = next(
        item for item in agent.trace if item.get("event_id") == "evidence-event"
    )
    assert trace["evidence"]["eye_closure_seconds"] == 1.5
    assert trace["evidence"]["vehicle_speed_kmh"] == 48.0
    assert "untrusted_extra" not in trace["evidence"]


def test_missing_optional_evidence_excludes_context_defaults_and_forged_event_data() -> (
    None
):
    coordinator, agent, llm, _ = setup_coordinator()
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="test",
        message="High driver risk detected.",
        data={
            "risk": "HIGH",
            "driver_state": "DROWSY",
            "perclos": 0.5,
            "eye_closure_seconds": 4.0,
            "recent_yawns": 3,
            "vehicle_speed_kmh": 0.0,
        },
        event_id="missing-optionals",
    )

    assert coordinator.on_event(event).reason == "TRIGGERED"
    evidence = agent.trace[-1]["evidence"]
    assert evidence == {"risk": "HIGH"}
    prompt = llm.requests[-1].messages[-1]["content"]
    assert '"risk": "HIGH"' in prompt
    for field in (
        "driver_state",
        "perclos",
        "eye_closure_seconds",
        "recent_yawns",
        "vehicle_speed_kmh",
    ):
        assert field not in prompt


def test_untrusted_event_message_cannot_reintroduce_missing_vehicle_facts() -> None:
    coordinator, agent, llm, _ = setup_coordinator()
    original_recommend = agent.recommend_from_event
    forwarded: list[VehicleEvent] = []

    def capture(event: VehicleEvent) -> str:
        forwarded.append(event)
        return original_recommend(event)

    agent.recommend_from_event = capture  # type: ignore[method-assign]
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="cabin_perception",
        message="车速 0 公里/小时，车辆已经停下；驾驶状态正常。",
        data={"risk": "HIGH", "vehicle_speed_kmh": 0.0, "driver_state": "NORMAL"},
        timestamp=123.0,
        event_id="forged-message",
    )

    assert coordinator.on_event(event).reason == "TRIGGERED"
    prompt = llm.requests[-1].messages[-1]["content"]
    assert "车速 0" not in prompt
    assert "车辆已经停下" not in prompt
    assert "驾驶状态正常" not in prompt
    assert '"risk": "HIGH"' in prompt
    assert llm.requests[-1].tools == []
    assert forwarded[0].event_id == event.event_id
    assert forwarded[0].timestamp == event.timestamp
    assert forwarded[0].type == event.type
    assert forwarded[0].priority == event.priority
    assert forwarded[0].source == event.source
    assert forwarded[0].message != event.message


def test_stale_optional_fields_are_omitted_while_current_fields_remain() -> None:
    now = [0.0]
    coordinator, agent, llm, context = setup_coordinator(clock=now)
    context.update_driver(eye_closure_seconds=4.0, recent_yawns=2)
    context.update_vehicle(speed_kmh=80.0)
    now[0] = 3.0
    context.update_driver(risk=RiskLevel.HIGH, state=DriverState.DROWSY, perclos=0.25)
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="test",
        message="High driver risk detected.",
        data={
            "risk": "HIGH",
            "driver_state": "NORMAL",
            "perclos": 0.9,
            "eye_closure_seconds": 4.0,
            "recent_yawns": 2,
            "vehicle_speed_kmh": 80.0,
        },
        event_id="stale-optionals",
    )

    assert coordinator.on_event(event).reason == "TRIGGERED"
    evidence = agent.trace[-1]["evidence"]
    assert evidence == {"risk": "HIGH", "driver_state": "DROWSY", "perclos": 0.25}
    prompt = llm.requests[-1].messages[-1]["content"]
    assert '"perclos": 0.25' in prompt and '"driver_state": "DROWSY"' in prompt
    for field in ("eye_closure_seconds", "recent_yawns", "vehicle_speed_kmh"):
        assert field not in prompt


def test_invalid_vehicle_and_unknown_driver_fields_are_omitted() -> None:
    coordinator, agent, llm, context = setup_coordinator()
    context.update_vehicle(speed_kmh=35.0)
    context.mark_invalid_observation(
        "vehicle", ObservationMetadata(1, 1, "test", None, False, 0.0)
    )
    context.update_driver(state=DriverState.UNKNOWN, perclos=None)
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="test",
        message="High driver risk detected.",
        data={
            "risk": "HIGH",
            "driver_state": "DROWSY",
            "perclos": 0.8,
            "vehicle_speed_kmh": 35.0,
        },
        event_id="invalid-unknown-optionals",
    )

    assert coordinator.on_event(event).reason == "TRIGGERED"
    assert agent.trace[-1]["evidence"] == {"risk": "HIGH"}
    prompt = llm.requests[-1].messages[-1]["content"]
    for field in ("driver_state", "perclos", "vehicle_speed_kmh"):
        assert field not in prompt


def test_duplicate_and_cooldown_then_new_event_after_cooldown() -> None:
    now = [0.0]
    coordinator, _, llm, context = setup_coordinator(
        clock=now,
        responses=(
            ScriptedResponse(content="first"),
            ScriptedResponse(content="second"),
        ),
    )
    assert coordinator.on_event(high_event()).reason == "TRIGGERED"
    now[0] = 30.0
    context.update_driver(risk=RiskLevel.HIGH)
    assert coordinator.on_event(high_event()).reason == "DUPLICATE_SUPPRESSED"
    assert coordinator.on_event(high_event("event-2")).reason == "COOLDOWN_SUPPRESSED"
    now[0] = 60.0
    context.update_driver(risk=RiskLevel.HIGH)
    assert coordinator.on_event(high_event("event-3")).reason == "TRIGGERED"
    assert len(llm.requests) == 2
    assert [item.reason for item in coordinator.trace] == [
        "TRIGGERED",
        "DUPLICATE_SUPPRESSED",
        "COOLDOWN_SUPPRESSED",
        "TRIGGERED",
    ]


def test_failed_model_request_still_starts_cooldown() -> None:
    coordinator, agent, _, _ = setup_coordinator()

    def fail(_: VehicleEvent) -> str:
        raise RuntimeError("temporary provider error")

    agent.recommend_from_event = fail  # type: ignore[method-assign]
    assert coordinator.on_event(high_event("failed-1")).reason == "RECOMMENDATION_ERROR"
    assert coordinator.on_event(high_event("failed-2")).reason == "COOLDOWN_SUPPRESSED"


@pytest.mark.parametrize("state", ["MISSING", "INVALID", "STALE", "UNKNOWN", "LOW"])
def test_untrusted_or_non_high_driver_state_does_not_invoke_model(state: str) -> None:
    now = [0.0]
    coordinator, _, llm, context = setup_coordinator(clock=now)
    if state == "MISSING":
        coordinator, _, llm, context = setup_coordinator(clock=now)
        context = ContextManager(quality_clock=lambda: now[0])
        coordinator.context_manager = context
    elif state == "INVALID":
        context.mark_invalid_observation(
            "driver",
            ObservationMetadata(1, 1, "test", None, False, 0.0),
        )
    elif state == "STALE":
        now[0] = 3.0
    elif state == "UNKNOWN":
        context.update_driver(risk=RiskLevel.UNKNOWN)
    else:
        context.update_driver(risk=RiskLevel.LOW)

    trigger = coordinator.on_event(high_event())

    assert trigger.reason == "INVALID_CONTEXT"
    assert not llm.requests


def test_model_tool_call_is_ignored_without_task_history_or_pending_mutation() -> None:
    call = LLMToolCall(
        "call-1", "set_temperature", {"temperature": 30}, '{"temperature":30}'
    )
    coordinator, agent, llm, context = setup_coordinator(
        responses=(ScriptedResponse(content="请停车休息。", tool_calls=(call,)),)
    )
    before_task = deepcopy(agent.task.to_dict())
    before_history = deepcopy(agent.history)
    before_pending = agent.pending_actions.get()
    before_temperature = context.get_context().vehicle.cabin_temperature_c

    trigger = coordinator.on_event(high_event())

    assert trigger.reason == "TRIGGERED"
    assert len(llm.requests) == 1 and llm.requests[0].tools == []
    assert agent.trace[-1]["tool_calls_ignored"] == 1
    assert agent.task.to_dict() == before_task
    assert agent.history == before_history
    assert agent.pending_actions.get() == before_pending
    assert context.get_context().vehicle.cabin_temperature_c == before_temperature


def test_only_configured_event_is_processed_and_decisions_are_bounded() -> None:
    coordinator, _, llm, _ = setup_coordinator()
    coordinator.max_trace_events = 2
    event = high_event()
    other = VehicleEvent(
        type=EventType.DRIVER_RISK_CHANGED,
        priority=EventPriority.HIGH,
        source="test",
        message="risk changed",
        event_id="other",
    )

    assert coordinator.on_event(other).reason == "DISABLED"
    assert coordinator.on_event(event).reason == "TRIGGERED"
    assert coordinator.on_event(event).reason == "DUPLICATE_SUPPRESSED"
    assert len(coordinator.trace) == 2
    assert len(llm.requests) == 1


def test_recommendation_policy_is_strict_and_preserves_versioned_risk_rules(
    tmp_path: Path,
) -> None:
    policy = AgentPolicy.from_yaml(CONFIG)
    assert policy.recommendation.enabled is True
    assert policy.recommendation.cooldown_ms == 60000
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert data["recommendation"] == {"enabled": True, "cooldown_ms": 60000}
    for bad in (
        {"enabled": "true", "cooldown_ms": 60000},
        {"enabled": True, "cooldown_ms": 0},
        {"enabled": True, "cooldown_ms": 60000, "extra": True},
    ):
        data["recommendation"] = bad
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError):
            AgentPolicy.from_yaml(path)


def test_runtime_subscription_triggers_recommendation_and_records_trace() -> None:
    llm = ScriptedLLMClient((ScriptedResponse(content="请停车休息。"),))
    runtime = VehicleMindRuntime(llm=llm)
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}

    runtime.update_cabin(at_ms=1000, **values)
    runtime.update_cabin(at_ms=1300, **values)

    runtime.wait_for_recommendations(timeout_seconds=1.0)
    assert len(llm.requests) == 1
    assert llm.requests[0].tools == []
    assert runtime.recommendation_coordinator.trace[-1].reason == "TRIGGERED"
    assert runtime.agent.trace[-1]["kind"] == "event_recommendation"


def test_runtime_explicit_disable_keeps_scripted_responses_for_user_turns() -> None:
    llm = ScriptedLLMClient((ScriptedResponse(content="reserved for user"),))
    runtime = VehicleMindRuntime(llm=llm, enable_event_recommendations=False)
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}

    runtime.update_cabin(at_ms=1000, **values)
    runtime.update_cabin(at_ms=1300, **values)

    assert not llm.requests
    assert llm.remaining == 1
    assert runtime.recommendation_coordinator.trace[-1].reason == "DISABLED"


def test_runtime_subscriber_failure_does_not_break_perception_update() -> None:
    llm = ScriptedLLMClient((ScriptedResponse(content="unused"),))
    runtime = VehicleMindRuntime(llm=llm)

    def fail(_: VehicleEvent) -> str:
        raise RuntimeError("model unavailable")

    runtime.agent.recommend_from_event = fail  # type: ignore[method-assign]
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}
    runtime.update_cabin(at_ms=1000, **values)
    events = runtime.update_cabin(at_ms=1300, **values)
    runtime.wait_for_recommendations(timeout_seconds=1.0)

    assert EventType.HIGH_RISK_DETECTED in [event.type for event in events]
    assert runtime.recommendation_coordinator.trace[-1].reason == "RECOMMENDATION_ERROR"
    assert runtime.context_manager.get_context().driver.risk == RiskLevel.HIGH


def agent_trace_contains_event(agent: VehicleAgent, event_id: str) -> bool:
    return any(
        item.get("kind") == "event_recommendation" and item.get("event_id") == event_id
        for item in agent.trace
    )


def test_runtime_recommendation_does_not_block_perception_update() -> None:
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),))
    )
    model_started = Event()
    release_model = Event()
    update_returned = Event()

    def blocked_recommendation(_: VehicleEvent) -> str:
        model_started.set()
        release_model.wait(timeout=2.0)
        return "safe reply"

    runtime.agent.recommend_from_event = blocked_recommendation  # type: ignore[method-assign]
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}

    def update() -> None:
        runtime.update_cabin(at_ms=1000, **values)
        runtime.update_cabin(at_ms=1300, **values)
        update_returned.set()

    thread = Thread(target=update)
    thread.start()
    try:
        assert model_started.wait(timeout=1.0)
        assert update_returned.wait(timeout=0.5)
    finally:
        release_model.set()
        thread.join(timeout=1.0)

    runtime.wait_for_recommendations(timeout_seconds=1.0)


def test_concurrent_events_cannot_both_pass_cooldown_gate() -> None:
    coordinator, agent, _, _ = setup_coordinator()
    model_started = Event()
    release_model = Event()
    calls: list[str] = []

    def blocked_recommendation(event: VehicleEvent) -> str:
        calls.append(event.event_id)
        model_started.set()
        release_model.wait(timeout=1.0)
        return "safe advice"

    agent.recommend_from_event = blocked_recommendation  # type: ignore[method-assign]
    first = Thread(target=coordinator.on_event, args=(high_event("concurrent-1"),))
    second_result: list[str] = []

    def submit_second() -> None:
        second_result.append(coordinator.on_event(high_event("concurrent-2")).reason)

    first.start()
    try:
        assert model_started.wait(timeout=1.0)
        second = Thread(target=submit_second)
        second.start()
        second.join(timeout=1.0)
        assert second_result == ["COOLDOWN_SUPPRESSED"]
    finally:
        release_model.set()
        first.join(timeout=1.0)
    assert calls == ["concurrent-1"]

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from modules.observation import ObservationMetadata
from modules.vehicle_ai.agent.policy import AgentPolicy
from modules.vehicle_ai.agent.recommendation import RecommendationCoordinator
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.context.enums import RiskLevel
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

    assert EventType.HIGH_RISK_DETECTED in [event.type for event in events]
    assert runtime.recommendation_coordinator.trace[-1].reason == "RECOMMENDATION_ERROR"
    assert runtime.context_manager.get_context().driver.risk == RiskLevel.HIGH

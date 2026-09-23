from __future__ import annotations

import pytest

from modules.vehicle_ai.context.models import VehicleContext
from modules.vehicle_ai.events import EventDetector, EventType
from modules.vehicle_ai.events.temporal_gate import TemporalGate
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.observation import ObservationMetadata


def test_gate_requires_sustained_observations_and_clears_on_recovery() -> None:
    gate = TemporalGate(hold_ms=300, cooldown_ms=1000)

    assert gate.observe(True, at_ms=0) is False
    assert gate.observe(False, at_ms=200) is False
    assert gate.observe(True, at_ms=300) is False
    assert gate.observe(True, at_ms=599) is False
    assert gate.observe(True, at_ms=600) is True


def test_gate_cooldown_suppresses_retrigger_until_window_ends() -> None:
    gate = TemporalGate(hold_ms=100, cooldown_ms=1000)

    assert gate.observe(True, at_ms=0) is False
    assert gate.observe(True, at_ms=100) is True
    assert gate.observe(False, at_ms=200) is False
    assert gate.observe(True, at_ms=300) is False
    assert gate.observe(True, at_ms=400) is False
    assert gate.observe(True, at_ms=1100) is True


def test_large_observation_gap_does_not_count_as_continuous_hazard() -> None:
    gate = TemporalGate(hold_ms=300, cooldown_ms=1000, max_gap_ms=500)

    assert gate.observe(True, at_ms=0) is False
    assert gate.observe(True, at_ms=1000) is False
    assert gate.observe(True, at_ms=1300) is True


def test_road_loss_requires_prior_positive_observation_and_hold() -> None:
    detector = EventDetector()
    context = VehicleContext()

    assert detector.observe_hazards("road", context, at_ms=0) == []
    context.road.lane_detected = True
    assert detector.observe_hazards("road", context, at_ms=100) == []
    context.road.lane_detected = False
    assert detector.observe_hazards("road", context, at_ms=200) == []
    events = detector.observe_hazards("road", context, at_ms=700)

    assert [event.type for event in events] == [EventType.LANE_LOST]


def test_unknown_lane_observation_cannot_be_coerced_into_loss() -> None:
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),))
    )
    runtime.update_driving(at_ms=0, lane_detected=True)

    with pytest.raises(TypeError):
        runtime.update_driving(at_ms=100, lane_detected=None)

    assert runtime.context_manager.get_context().road.lane_detected is True
    assert runtime.update_driving(at_ms=700, lane_detected=True) == []


def test_driver_high_risk_requires_second_observation() -> None:
    detector = EventDetector()
    context = VehicleContext()
    context.driver.risk = "HIGH"

    assert detector.observe_hazards("driver", context, at_ms=1000) == []
    events = detector.observe_hazards("driver", context, at_ms=1300)

    assert [event.type for event in events] == [
        EventType.DRIVER_RISK_CHANGED,
        EventType.HIGH_RISK_DETECTED,
    ]


def test_runtime_evaluates_hazard_when_context_values_do_not_change() -> None:
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),))
    )
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}

    first = runtime.update_cabin(at_ms=1000, **values)
    second = runtime.update_cabin(at_ms=1300, **values)

    assert EventType.HIGH_RISK_DETECTED not in [event.type for event in first]
    assert EventType.DRIVER_RISK_CHANGED not in [event.type for event in first]
    assert [event.type for event in second] == [
        EventType.DRIVER_RISK_CHANGED,
        EventType.HIGH_RISK_DETECTED,
    ]


def test_invalid_observation_does_not_advance_high_risk_hold() -> None:
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),))
    )
    values = {"presence": "PRESENT", "driver_state": "DROWSY", "risk": "HIGH"}
    invalid = ObservationMetadata(
        timestamp_ms=1250,
        sequence=1,
        source="cabin_perception",
        confidence=None,
        valid=False,
        processing_ms=1.0,
    )

    runtime.update_cabin(at_ms=1000, **values)
    assert runtime.update_cabin(metadata=invalid, at_ms=1300, **values) == []
    assert EventType.HIGH_RISK_DETECTED not in [
        event.type for event in runtime.update_cabin(at_ms=1400, **values)
    ]
    assert EventType.HIGH_RISK_DETECTED in [
        event.type for event in runtime.update_cabin(at_ms=1700, **values)
    ]

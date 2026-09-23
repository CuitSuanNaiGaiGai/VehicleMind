from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from modules.vehicle_ai.replay import load_replay_scenario
from modules.vehicle_ai.replay.models import (
    ExpectedOutcome,
    ReplayObservation,
    ReplayStep,
    ScriptedResponse,
    freeze_mapping,
)
from modules.vehicle_ai.replay.runner import ReplayRunner
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.replay.trace import TraceRecorder
from modules.vehicle_ai.runtime import VehicleMindRuntime


ROOT = Path(__file__).resolve().parents[3]
SCENARIO_PATH = ROOT / "assets/scenarios/drowsy_rest_stop.yaml"


def _scenario():
    return load_replay_scenario(SCENARIO_PATH, repository_root=ROOT)


def test_showcase_replay_completes_the_perception_to_action_loop() -> None:
    result = ReplayRunner().run(_scenario())

    assert result.passed is True
    assert result.context_schema_version == 1
    assert len(result.semantic_sha256) == 64
    assert result.final_context["driver"]["state"] == "DROWSY"
    assert result.final_context["driver"]["risk"] == "HIGH"
    assert result.final_context["vehicle"]["navigation_state"] == "ACTIVE"
    assert result.final_context["vehicle"]["navigation_destination_id"] == (
        "rest_area_001"
    )
    assert "HIGH_RISK_DETECTED" in result.event_types
    assert result.successful_tools == (
        "search_nearby_rest_area",
        "start_navigation",
    )
    assert result.unauthorized_sensitive_executions == 0
    assert result.remaining_scripted_responses == 0
    assert set(result.metrics) == {
        "total_ms",
        "context_update_ms",
        "agent_ms",
        "confirmation_ms",
    }
    assert all(assertion.passed for assertion in result.assertions)


def test_semantic_digest_is_stable_across_replays() -> None:
    first = ReplayRunner().run(_scenario())
    second = ReplayRunner().run(_scenario())

    assert first.semantic_sha256 == second.semantic_sha256
    assert first.final_context == second.final_context


def test_trace_records_observation_provenance_and_unknown_evidence() -> None:
    result = ReplayRunner().run(_scenario())
    updates = [record for record in result.trace if record.kind == "context_update"]

    assert any(
        record.data["domain"] == "cabin"
        and record.data["source"] == "recorded_cabin_perception"
        and record.data["confidence"] is None
        and record.data["values"]["eye_closed"] is None
        for record in updates
    )


def test_mismatched_expectation_returns_named_failure() -> None:
    scenario = _scenario()
    wrong = ExpectedOutcome(
        event_types=scenario.expected.event_types,
        successful_tools=scenario.expected.successful_tools,
        final_vehicle=freeze_mapping({"navigation_state": "IDLE"}),
        unauthorized_sensitive_executions=0,
    )

    result = ReplayRunner().run(replace(scenario, expected=wrong))

    assert result.passed is False
    failures = {item.name: item for item in result.assertions if not item.passed}
    assert "vehicle.navigation_state" in failures
    assert failures["vehicle.navigation_state"].actual == "ACTIVE"


def test_invalid_observation_is_recorded_but_not_applied() -> None:
    scenario = _scenario()
    invalid = replace(scenario.steps[-1], at_ms=3200, confirm_pending=False)
    invalid_cabin = replace(
        scenario.steps[0].cabin,
        valid=False,
        confidence=0.1,
        values=freeze_mapping({"driver_state": "NORMAL", "risk": "LOW"}),
    )
    invalid = replace(invalid, cabin=invalid_cabin)
    modified = replace(scenario, steps=(*scenario.steps, invalid))

    result = ReplayRunner().run(modified)

    assert result.final_context["driver"]["state"] == "DROWSY"
    record = [item for item in result.trace if item.at_ms == 3200][0]
    assert record.data["valid"] is False
    assert record.data["applied"] is False


def test_invalid_replay_frame_breaks_high_risk_hold() -> None:
    scenario = _scenario()
    invalid_cabin = replace(scenario.steps[2].cabin, valid=False)
    invalid_step = ReplayStep(at_ms=1950, cabin=invalid_cabin)
    modified = replace(
        scenario,
        steps=(*scenario.steps[:3], invalid_step, *scenario.steps[3:]),
    )

    result = ReplayRunner().run(modified)

    assert "HIGH_RISK_DETECTED" not in result.event_types


def test_valid_vehicle_replay_observation_keeps_metadata() -> None:
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),))
    )
    observation = ReplayObservation(
        source="recorded_vehicle_state",
        confidence=1.0,
        valid=True,
        values=freeze_mapping({"speed_kmh": 30.0}),
    )

    ReplayRunner()._apply_observation(
        runtime,
        TraceRecorder(),
        at_ms=100,
        sequence=0,
        domain="vehicle",
        observation=observation,
    )

    quality = runtime.context_manager.observation_quality("vehicle")
    assert quality["metadata"].source == "recorded_vehicle_state"
    assert quality["status"] == "KNOWN"

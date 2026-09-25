from __future__ import annotations

import pytest

from modules.observation import ObservationMetadata
from modules.vehicle_ai.context import ContextManager, DriverState, RiskLevel
from modules.vehicle_ai.context.context_selector import ContextSelector
from modules.vehicle_ai.events import EventDetector
from modules.vehicle_ai.integration.cabin_adapter import CabinContextAdapter
from modules.vehicle_ai.runtime import VehicleMindRuntime


def test_default_false_is_missing_until_a_road_observation_arrives() -> None:
    manager = ContextManager()

    assert manager.field_quality("road", "lane_detected", now=10.0) == "MISSING"
    assert manager.freshness()["road"]["fresh"] is False
    manager.update_road(lane_detected=False)
    received_at = manager.observation_quality("road")["received_at"]

    assert manager.field_quality("road", "lane_detected", now=received_at) == "KNOWN"
    assert manager.get_context().road.lane_detected is False


def test_unknown_is_not_the_same_as_missing() -> None:
    manager = ContextManager()
    manager.update_driver(state=DriverState.UNKNOWN)
    received_at = manager.observation_quality("driver")["received_at"]

    assert manager.field_quality("driver", "state", now=received_at) == "UNKNOWN"


def test_partial_update_does_not_mark_untouched_defaults_as_observed() -> None:
    manager = ContextManager()
    manager.update_road(vehicle_count=1)

    assert manager.field_quality("road", "vehicle_count") == "KNOWN"
    assert manager.field_quality("road", "lane_detected") == "MISSING"


def test_cabin_adapter_only_receipts_explicit_fields_including_zero() -> None:
    manager = ContextManager()
    adapter = CabinContextAdapter(manager)

    adapter.update(presence="PRESENT", driver_state="DROWSY", risk="HIGH")

    assert manager.field_quality("driver", "risk") == "KNOWN"
    for field in (
        "perclos",
        "eye_closed",
        "eye_closure_seconds",
        "recent_yawns",
        "blink_count",
    ):
        assert manager.field_quality("driver", field) == "MISSING"
    assert manager.get_context().driver.eye_closure_seconds == 0.0
    assert manager.get_context().driver.recent_yawns == 0
    assert manager.field_quality("vehicle", "speed_kmh") == "MISSING"

    adapter.update(
        presence="PRESENT",
        driver_state="DROWSY",
        risk="HIGH",
        eye_closure_seconds=0.0,
        recent_yawns=0,
        blink_count=0,
    )
    for field in ("eye_closure_seconds", "recent_yawns", "blink_count"):
        assert manager.field_quality("driver", field) == "KNOWN"
    manager.update_vehicle(speed_kmh=0.0)
    assert manager.field_quality("vehicle", "speed_kmh") == "KNOWN"


def test_selector_excludes_unobserved_fields_after_partial_update() -> None:
    manager = ContextManager()
    manager.update_vehicle(speed_kmh=37)
    manager.update_driver(risk=RiskLevel.HIGH)
    selected = (
        ContextSelector()
        .select(
            "驾驶员状态和车速如何？",
            manager.get_context(),
            driver_quality=manager.observation_quality("driver")["status"],
            vehicle_quality=manager.observation_quality("vehicle")["status"],
            field_quality=manager.field_quality,
        )
        .context
    )
    assert selected["vehicle"]["speed_kmh"] == 37
    assert "gear" not in selected["vehicle"]
    assert "state" not in selected["driver"]


def test_stale_uses_receipt_clock_and_domain_ttl() -> None:
    manager = ContextManager()
    manager.update_road(lane_detected=True)
    received_at = manager.observation_quality("road")["received_at"]

    assert (
        manager.field_quality("road", "lane_detected", now=received_at + 1) == "KNOWN"
    )
    assert (
        manager.field_quality("road", "lane_detected", now=received_at + 1.001)
        == "STALE"
    )


def test_failed_semantic_update_does_not_mark_missing_domain_as_observed() -> None:
    manager = ContextManager()

    with pytest.raises(TypeError):
        manager.update_road(lane_detected="no")

    assert manager.field_quality("road", "lane_detected") == "MISSING"


def test_invalid_perception_does_not_replace_last_semantic_value() -> None:
    runtime = VehicleMindRuntime.__new__(VehicleMindRuntime)
    runtime.context_manager = ContextManager()
    runtime.driving = None
    runtime.event_detector = EventDetector()
    runtime.event_bus = None
    metadata = ObservationMetadata(
        timestamp_ms=50,
        sequence=0,
        source="driving_perception",
        confidence=None,
        valid=False,
        processing_ms=2.0,
    )

    assert runtime.update_driving(metadata=metadata, lane_detected=False) == []
    assert runtime.context_manager.field_quality("road", "lane_detected") == "INVALID"
    assert runtime.context_manager.get_context().road.lane_detected is False


def test_valid_perception_metadata_is_available_in_quality_report() -> None:
    manager = ContextManager()
    metadata = ObservationMetadata(
        timestamp_ms=123,
        sequence=4,
        source="driving_perception",
        confidence=None,
        valid=True,
        processing_ms=3.0,
    )
    manager.update_road(observation=metadata, lane_detected=False)

    report = manager.observation_quality("road")
    assert report["metadata"] == metadata
    assert report["status"] == "KNOWN"

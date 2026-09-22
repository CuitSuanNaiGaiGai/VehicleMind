from __future__ import annotations

import pytest

from modules.observation import ObservationMetadata
from modules.vehicle_ai.context import ContextManager, DriverState
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
    runtime.event_detector = None
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
    manager.update_road(lane_detected=False)

    manager.mark_valid_observation("road", metadata)

    report = manager.observation_quality("road")
    assert report["metadata"] == metadata
    assert report["status"] == "KNOWN"

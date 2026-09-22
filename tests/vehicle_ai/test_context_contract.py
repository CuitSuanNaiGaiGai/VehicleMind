from __future__ import annotations

from dataclasses import fields

import pytest

from modules.vehicle_ai.context import ContextManager, DriverContext, RoadContext
from modules.vehicle_ai.context import VehicleStatus
from modules.vehicle_ai.context.contract import (
    CONTEXT_FIELD_CONTRACTS,
    CONTEXT_SCHEMA_VERSION,
)


def test_versioned_contract_covers_every_public_domain_field() -> None:
    for domain, model in (
        ("driver", DriverContext),
        ("road", RoadContext),
        ("vehicle", VehicleStatus),
    ):
        expected = {item.name for item in fields(model)} - {"updated_at", "source"}
        assert set(CONTEXT_FIELD_CONTRACTS[domain]) == expected

    assert ContextManager().get_context().to_dict()["schema_version"] == (
        CONTEXT_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    ("domain", "values", "error"),
    [
        ("driver", {"eye_closed": 0}, TypeError),
        ("driver", {"perclos": float("nan")}, ValueError),
        ("driver", {"perclos": 1.01}, ValueError),
        ("driver", {"recent_yawns": -1}, ValueError),
        ("driver", {"state": "DROWSY"}, TypeError),
        ("road", {"vehicle_count": True}, TypeError),
        ("road", {"lane_detected": None}, TypeError),
        ("road", {"traffic_level": "CONGESTED"}, ValueError),
        ("vehicle", {"speed_kmh": float("inf")}, ValueError),
        ("vehicle", {"volume": 101}, ValueError),
        ("vehicle", {"gear": "D"}, TypeError),
    ],
)
def test_invalid_update_preserves_context_timestamp_and_history(
    domain: str,
    values: dict[str, object],
    error: type[Exception],
) -> None:
    manager = ContextManager()
    before = manager.get_context()
    before_history = manager.recent_changes()
    update = getattr(manager, f"update_{domain}")

    with pytest.raises(error):
        update(**values)

    assert manager.get_context() == before
    assert manager.recent_changes() == before_history


def test_late_invalid_field_does_not_partially_commit() -> None:
    manager = ContextManager()
    before = manager.get_context()

    with pytest.raises(ValueError, match="perclos"):
        manager.update_driver(recent_yawns=2, perclos=2.0)

    assert manager.get_context() == before
    assert manager.recent_changes() == []


def test_unknown_or_internal_field_does_not_partially_commit() -> None:
    manager = ContextManager()
    before = manager.get_context()

    with pytest.raises(AttributeError, match="missing"):
        manager.update_road(vehicle_count=2, missing=1)
    with pytest.raises(ValueError, match="updated_at"):
        manager.update_road(vehicle_count=2, updated_at=42.0)

    assert manager.get_context() == before
    assert manager.recent_changes() == []


def test_valid_boundaries_preserve_unknown_and_false() -> None:
    manager = ContextManager()

    manager.update_driver(perclos=1.0, eye_closed=None)
    manager.update_road(lane_detected=False, traffic_level="UNKNOWN")
    manager.update_vehicle(volume=0, speed_kmh=0.0)

    context = manager.get_context()
    assert context.driver.perclos == 1.0
    assert context.driver.eye_closed is None
    assert context.road.lane_detected is False
    assert context.road.traffic_level == "UNKNOWN"
    assert context.vehicle.volume == 0

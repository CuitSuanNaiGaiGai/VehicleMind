from __future__ import annotations

import pytest

from modules.config.events import EventTimingConfig, HazardTiming


def test_versioned_event_timing_file_loads() -> None:
    config = EventTimingConfig.load()

    assert config.schema_version == 1
    assert config.high_driver_risk.hold_ms == 300
    assert config.lane_lost.cooldown_ms == 10000


@pytest.mark.parametrize("value", [True, -1, 1.2])
def test_hold_duration_requires_nonnegative_integer(value: object) -> None:
    with pytest.raises(ValueError):
        HazardTiming(hold_ms=value, cooldown_ms=100)


def test_gate_rejects_time_regression() -> None:
    from modules.vehicle_ai.events.temporal_gate import TemporalGate

    gate = TemporalGate(hold_ms=100, cooldown_ms=1000)
    gate.observe(True, at_ms=200)

    with pytest.raises(ValueError, match="must not go backwards"):
        gate.observe(True, at_ms=199)

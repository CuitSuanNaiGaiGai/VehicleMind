from __future__ import annotations

from apps.vehicle_ai_demo.quality_display import format_age


def test_missing_observation_age_displays_as_unavailable() -> None:
    assert format_age(None) == "未观测"


def test_observed_age_keeps_seconds_precision() -> None:
    assert format_age(1.23456) == "1.235 秒"

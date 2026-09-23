from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class HazardTiming:
    hold_ms: int
    cooldown_ms: int
    max_gap_ms: int | None = None

    def __post_init__(self) -> None:
        if type(self.hold_ms) is not int or self.hold_ms < 0:
            raise ValueError("hold_ms must be a nonnegative integer")
        if type(self.cooldown_ms) is not int or self.cooldown_ms <= 0:
            raise ValueError("cooldown_ms must be a positive integer")
        if self.max_gap_ms is not None and (
            type(self.max_gap_ms) is not int or self.max_gap_ms <= 0
        ):
            raise ValueError("max_gap_ms must be a positive integer")


@dataclass(frozen=True)
class EventTimingConfig:
    high_driver_risk: HazardTiming
    lane_lost: HazardTiming
    drivable_area_lost: HazardTiming
    schema_version: int = 1

    @classmethod
    def load(cls, path: Path | None = None) -> EventTimingConfig:
        config_path = path or Path(__file__).with_suffix(".yaml")
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "high_driver_risk",
            "lane_lost",
            "drivable_area_lost",
        }:
            raise ValueError("event timing configuration has invalid fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported event timing schema version")
        timing = {}
        for name in ("high_driver_risk", "lane_lost", "drivable_area_lost"):
            values = data[name]
            if not isinstance(values, dict) or set(values) != {
                "hold_ms",
                "cooldown_ms",
                "max_gap_ms",
            }:
                raise ValueError(f"{name} timing has invalid fields")
            timing[name] = HazardTiming(**values)
        return cls(
            high_driver_risk=timing["high_driver_risk"],
            lane_lost=timing["lane_lost"],
            drivable_area_lost=timing["drivable_area_lost"],
        )

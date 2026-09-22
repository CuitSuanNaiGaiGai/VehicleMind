from __future__ import annotations

import math

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from modules.vehicle_ai.context.enums import (
    DriverPresence,
    DriverState,
    GearState,
    NavigationState,
    RiskLevel,
)


CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FieldContract:
    kind: str
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    enum_type: type[Enum] | None = None
    choices: frozenset[str] | None = None


CONTEXT_FIELD_CONTRACTS: Mapping[str, Mapping[str, FieldContract]] = {
    "driver": {
        "presence": FieldContract("enum", enum_type=DriverPresence),
        "state": FieldContract("enum", enum_type=DriverState),
        "risk": FieldContract("enum", enum_type=RiskLevel),
        "perclos": FieldContract("number", nullable=True, minimum=0, maximum=1),
        "eye_closed": FieldContract("boolean", nullable=True),
        "eye_closure_seconds": FieldContract("number", minimum=0),
        "recent_yawns": FieldContract("integer", minimum=0),
        "blink_count": FieldContract("integer", minimum=0),
    },
    "road": {
        "vehicle_count": FieldContract("integer", minimum=0),
        "pedestrian_count": FieldContract("integer", minimum=0),
        "rider_count": FieldContract("integer", minimum=0),
        "traffic_light_count": FieldContract("integer", minimum=0),
        "traffic_sign_count": FieldContract("integer", minimum=0),
        "total_objects": FieldContract("integer", minimum=0),
        "lane_detected": FieldContract("boolean"),
        "drivable_area_detected": FieldContract("boolean"),
        "traffic_level": FieldContract(
            "string",
            choices=frozenset({"UNKNOWN", "LIGHT", "MODERATE", "HEAVY"}),
        ),
    },
    "vehicle": {
        "speed_kmh": FieldContract("number", minimum=0, maximum=400),
        "gear": FieldContract("enum", enum_type=GearState),
        "cabin_temperature_c": FieldContract("number", minimum=-40, maximum=85),
        "target_temperature_c": FieldContract("number", minimum=16, maximum=30),
        "ac_enabled": FieldContract("boolean"),
        "driver_window_open": FieldContract("boolean"),
        "passenger_window_open": FieldContract("boolean"),
        "media_playing": FieldContract("boolean"),
        "media_title": FieldContract("string", nullable=True),
        "volume": FieldContract("integer", minimum=0, maximum=100),
        "navigation_state": FieldContract("enum", enum_type=NavigationState),
        "navigation_destination_id": FieldContract("string", nullable=True),
        "navigation_destination": FieldContract("string", nullable=True),
    },
}


def _validate_field(name: str, value: object, rule: FieldContract) -> None:
    if value is None:
        if rule.nullable:
            return
        raise TypeError(f"{name} cannot be null")

    numeric_value: float | None = None
    if rule.kind == "boolean":
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
    elif rule.kind == "integer":
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
        numeric_value = float(value)
    elif rule.kind == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        numeric_value = float(value)
    elif rule.kind == "enum":
        enum_type = rule.enum_type
        if enum_type is None:
            raise RuntimeError(f"{name} has no enum type in its contract")
        if not isinstance(value, enum_type):
            raise TypeError(f"{name} must be a {enum_type.__name__}")
    elif rule.kind == "string":
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if rule.choices is not None and value not in rule.choices:
            raise ValueError(f"{name} must be one of {sorted(rule.choices)}")
    else:
        raise RuntimeError(f"unsupported field contract kind: {rule.kind}")

    if numeric_value is not None:
        if rule.minimum is not None and numeric_value < rule.minimum:
            raise ValueError(f"{name} must be at least {rule.minimum}")
        if rule.maximum is not None and numeric_value > rule.maximum:
            raise ValueError(f"{name} must be at most {rule.maximum}")


def validate_domain_updates(domain: str, updates: Mapping[str, object]) -> None:
    if domain not in CONTEXT_FIELD_CONTRACTS:
        raise ValueError(f"unknown context domain: {domain}")
    rules = CONTEXT_FIELD_CONTRACTS[domain]
    for field_name, value in updates.items():
        if field_name in {"updated_at", "source"}:
            raise ValueError(f"Field cannot be updated directly: {field_name}")
        if field_name not in rules:
            raise AttributeError(f"{domain} context has no field '{field_name}'")
        _validate_field(f"{domain}.{field_name}", value, rules[field_name])

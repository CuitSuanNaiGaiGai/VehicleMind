from __future__ import annotations

import math

import pytest

from modules.vehicle_ai.tools.validation import valid_arguments


SCHEMA = {
    "required": ["query", "distance_km"],
    "properties": {
        "query": {"type": "string", "enum": ["休息", "服务区"]},
        "distance_km": {"type": "number", "minimum": 1, "maximum": 50},
    },
}


def test_scalar_tool_arguments_match_schema() -> None:
    assert valid_arguments({"query": "休息", "distance_km": 8.5}, SCHEMA)
    assert not valid_arguments(["休息", 8.5], SCHEMA)
    assert not valid_arguments({"query": "休息"}, SCHEMA)
    assert not valid_arguments(
        {"query": "休息", "distance_km": 8.5, "unlisted": True}, SCHEMA
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, 0, 51, True, "12"])
def test_tool_schema_rejects_nonfinite_out_of_range_and_wrong_types(value) -> None:
    assert not valid_arguments({"query": "休息", "distance_km": value}, SCHEMA)


def test_tool_schema_enforces_enum_membership() -> None:
    assert not valid_arguments(
        {"query": "destination from user text", "distance_km": 8}, SCHEMA
    )

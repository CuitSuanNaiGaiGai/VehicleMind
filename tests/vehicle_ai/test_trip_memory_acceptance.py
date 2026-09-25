from __future__ import annotations

from modules.vehicle_ai.memory.event_store import TripEvent, TripEventStore
from modules.vehicle_ai.memory.tool import build_trip_memory_tool


def test_frozen_trip_queries_exactly_retrieve_expected_event_sets(tmp_path) -> None:
    store = TripEventStore(tmp_path / "events.sqlite3")
    fixtures = (
        TripEvent("risk-1", "trip-a", "RISK", 90, "event_bus", {"risk": "HIGH"}),
        TripEvent("rem-1", "trip-a", "REMINDER", 100, "agent", {"text": "休息"}),
        TripEvent("cancel-1", "trip-a", "ACTION_CANCELLED", 150, "user", {"poi": "P1"}),
        TripEvent("rem-2", "trip-a", "REMINDER", 200, "agent", {"text": "休息"}),
        TripEvent("other-trip", "trip-b", "REMINDER", 110, "agent", {}),
    )
    for event in fixtures:
        store.append(event)
    tool = build_trip_memory_tool(store, "trip-a")

    queries = (
        ({"event_type": "REMINDER"}, 2, ["rem-1", "rem-2"]),
        ({"event_type": "ACTION_CANCELLED"}, 1, ["cancel-1"]),
        ({"since": 140, "until": 160}, 1, ["cancel-1"]),
        ({}, 4, ["risk-1", "rem-1", "cancel-1", "rem-2"]),
    )
    correct = 0
    for arguments, expected_count, expected_ids in queries:
        result = tool.handler(**arguments)
        actual_ids = [event["event_id"] for event in result.data["events"]]
        correct += result.success and result.data["count"] == expected_count
        correct += actual_ids == expected_ids

    assert correct == len(queries) * 2

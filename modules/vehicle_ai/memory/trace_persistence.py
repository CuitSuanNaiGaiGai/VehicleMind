"""Translate Agent traces into compact trip-memory events."""

from __future__ import annotations

import time
import uuid
from typing import Any

from modules.vehicle_ai.memory.event_store import TripEvent


_READ_ONLY_TOOLS = {
    "search_vehicle_knowledge",
    "query_trip_events",
    "get_vehicle_status",
    "get_climate_status",
    "get_media_status",
    "search_nearby_rest_area",
}


def trace_to_trip_event(trip_id: str, trace: dict[str, Any]) -> TripEvent | None:
    """Return no event for non-persisted trace items and read-only tool calls."""
    kind = trace.get("kind")
    event_type: str | None = None
    if kind == "user_request":
        event_type = "USER_REQUEST"
    elif kind == "selection":
        event_type = "USER_SELECTION"
    elif kind == "confirmation":
        event_type = (
            "ACTION_CONFIRMED"
            if trace.get("result", {}).get("success")
            else "ACTION_OUTCOME"
        )
    elif kind == "rejection" and trace.get("result", {}).get("success"):
        event_type = "ACTION_CANCELLED"
    elif kind == "pending_expired":
        event_type = "ACTION_EXPIRED"
    elif kind == "plan_step":
        event_type = "TASK_STEP"
    elif kind == "tool_result":
        source = trace.get("source", "")
        if source in _READ_ONLY_TOOLS:
            return None
        event_type = (
            "ACTION_PROPOSED"
            if trace.get("result", {}).get("error") == "CONFIRMATION_REQUIRED"
            else "ACTION_OUTCOME"
        )
    if event_type is None:
        return None
    event_id = f"{trip_id}:trace:{uuid.uuid4().hex}"
    return TripEvent(
        event_id=event_id,
        trip_id=trip_id,
        event_type=event_type,
        occurred_at=float(trace.get("at_seconds", time.time())),
        source=str(trace.get("source", kind)),
        payload={key: value for key, value in trace.items() if key != "task"},
    )

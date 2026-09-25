"""Read-only Agent tool for querying the current trip's event history."""

from __future__ import annotations

import sqlite3

from modules.vehicle_ai.memory.event_store import TripEventStore
from modules.vehicle_ai.tools.base import ToolDefinition, ToolResult

EVENT_TYPES = (
    "RISK",
    "REMINDER",
    "USER_REQUEST",
    "USER_SELECTION",
    "ACTION_PROPOSED",
    "ACTION_CONFIRMED",
    "ACTION_CANCELLED",
    "ACTION_OUTCOME",
    "ACTION_EXPIRED",
)


def build_trip_memory_tool(
    store: TripEventStore, trip_id: str
) -> ToolDefinition:
    """Expose only bounded filters; trip identity is fixed by the runtime."""

    def query_trip_events(
        event_type: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 20,
    ) -> ToolResult:
        if event_type is not None and event_type not in EVENT_TYPES:
            return ToolResult(False, "事件类型不受支持。", error="INVALID_EVENT_TYPE")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            return ToolResult(False, "历史查询条数需为 1 至 50。", error="INVALID_LIMIT")
        if since is not None and not isinstance(since, int | float):
            return ToolResult(False, "起始时间必须为 Unix 秒时间戳。", error="INVALID_TIME")
        if until is not None and not isinstance(until, int | float):
            return ToolResult(False, "结束时间必须为 Unix 秒时间戳。", error="INVALID_TIME")
        try:
            events = store.query(
                trip_id=trip_id,
                event_type=event_type,
                since=since,
                until=until,
                limit=limit,
            )
            count = store.count(
                trip_id=trip_id,
                event_type=event_type,
                since=since,
                until=until,
            )
        except ValueError as error:
            return ToolResult(False, str(error), error="INVALID_QUERY")
        except (OSError, sqlite3.Error):
            return ToolResult(
                False,
                "行程历史暂不可用；当前车辆状态不受影响。",
                error="MEMORY_UNAVAILABLE",
            )
        serialized = [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "occurred_at": event.occurred_at,
                "source": event.source,
                "payload": event.payload,
            }
            for event in events
        ]
        return ToolResult(
            True,
            f"当前行程共检索到 {count} 条匹配历史事件。",
            {
                "scope": "trip_history_not_current_state",
                "trip_id": trip_id,
                "event_type": event_type,
                "count": count,
                "returned": len(serialized),
                "events": serialized,
            },
        )

    return ToolDefinition(
        name="query_trip_events",
        description=(
            "只读查询当前行程的历史事件，用于回答提醒次数、用户选择、取消和动作结果；"
            "这是历史记录，不代表车辆当前感知状态。时间使用 Unix 秒时间戳，结果最多 50 条。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
                "since": {"type": "number", "description": "起始 Unix 秒时间戳，含边界"},
                "until": {"type": "number", "description": "结束 Unix 秒时间戳，含边界"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=query_trip_events,
        category="memory",
        read_only=True,
    )

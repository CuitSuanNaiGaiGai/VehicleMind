from __future__ import annotations

import time

from modules.vehicle_ai.memory import TripEventStore


def handle_trip_memory_command(
    command: str, store: TripEventStore, trip_id: str
) -> bool:
    """Display or explicitly clear this trip's persisted history."""
    if command == "history":
        history = store.query(trip_id=trip_id, limit=20)
        if not history:
            print("本次行程暂无持久化事件。")
        else:
            for event in history:
                timestamp = time.strftime(
                    "%H:%M:%S", time.localtime(event.occurred_at)
                )
                print(
                    f"{timestamp} {event.event_type} · {event.source} · {event.payload}"
                )
        return True
    if command == "clear-history":
        confirmation = input(f"输入当前行程 ID（{trip_id}）以确认清空：").strip()
        if confirmation == trip_id:
            removed = store.clear_trip(trip_id)
            print(f"已清空本次行程历史，共删除 {removed} 条事件。")
        else:
            print("行程 ID 不匹配，未清空历史。")
        return True
    return False

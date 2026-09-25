"""Prompt construction for event-triggered, no-tools safety advice."""

from __future__ import annotations

import json

from modules.vehicle_ai.events import VehicleEvent


EVIDENCE_FIELDS = (
    "driver_state",
    "risk",
    "perclos",
    "eye_closure_seconds",
    "recent_yawns",
    "vehicle_speed_kmh",
)


def build_event_advice_messages(
    event: VehicleEvent,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Build a prompt from event text and explicitly approved evidence fields."""
    evidence = {key: event.data[key] for key in EVIDENCE_FIELDS if key in event.data}
    messages = [
        {
            "role": "system",
            "content": (
                "你是车载安全建议助手。请结合可信感知证据，给出简短、平和、可执行的安全建议。"
                "高风险疲劳时优先建议安全停车休息；不得声称音乐可以消除疲劳，也不得声称已执行任何动作。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"事件编号：{event.event_id}\n"
                f"事件说明：{event.message}\n"
                "白名单感知证据（JSON）："
                f"{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}\n"
                "请根据证据给出驾驶员此刻最合适的下一步建议；不要调用或声称调用车机工具。"
            ),
        },
    ]
    return messages, evidence

from __future__ import annotations

import json

from typing import Any

from modules.vehicle_ai.agent.action_state import PendingActionStore


def pending_action_message(store: PendingActionStore) -> dict:
    pending = store.to_agent_context()
    if pending is None:
        return {"role": "system", "content": "PENDING ACTION:\nNone"}

    pending_json = json.dumps(
        pending,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return {
        "role": "system",
        "content": (
            "PENDING ACTION:\n"
            f"{pending_json}\n\n"
            "If the user confirms this action, execute the exact "
            "tool_name with the exact stored arguments."
        ),
    }


def assistant_tool_message(response: Any) -> dict:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments_json,
                },
            }
            for call in response.tool_calls
        ],
    }


def print_pending_action(store: PendingActionStore) -> None:
    pending = store.get()
    if pending is None:
        print("[Pending Action] None")
        return

    print("[Pending Action]")
    print(
        json.dumps(
            pending.to_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

"""Build a focused, evidence-grounded vehicle context message for one turn."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from modules.vehicle_ai.agent.decision_context import (
    GROUNDING_NOTE,
    attach_field_evidence,
)

if TYPE_CHECKING:
    from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent


def build_context_message(
    agent: VehicleAgent, user_text: str, debug: bool = False
) -> dict:
    """Select relevant current context and attach quality/freshness evidence."""
    manager = agent.context_manager
    selection = agent.context_selector.select(
        user_text=user_text,
        vehicle_context=manager.get_context(),
        road_quality=manager.observation_quality("road")["status"],
        driver_quality=manager.observation_quality("driver")["status"],
        vehicle_quality=manager.observation_quality("vehicle")["status"],
        field_quality=manager.field_quality,
    )
    selected_context = attach_field_evidence(selection.context, manager)

    if debug:
        print("\n[Context Selector]")
        print("  topics:", [topic.value for topic in selection.topics])
        print("  matched:", selection.matched_keywords)
        print("  context:")
        print(
            json.dumps(
                selected_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    if not selected_context:
        return {
            "role": "system",
            "content": (
                "CURRENT RELEVANT VEHICLE CONTEXT:\n"
                "No vehicle context is required for this request."
            ),
        }

    context_json = json.dumps(
        selected_context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return {
        "role": "system",
        "content": (
            "CURRENT RELEVANT VEHICLE CONTEXT:\n"
            f"{context_json}\n\n"
            "Only use this context when it is relevant to the current user request. "
            "If a domain quality is STALE, INVALID or MISSING, prior turns and "
            "stored values do not establish current facts for that domain.\n"
            f"{GROUNDING_NOTE}"
        ),
    }

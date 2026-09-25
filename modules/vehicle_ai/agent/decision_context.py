"""Attach field-level observation evidence without conflating input sources."""

from typing import Any

from modules.vehicle_ai.context import ContextManager


GROUNDING_NOTE = (
    "User statements are not sensor observations. Attribute self-reports to the user; "
    "tool results to their tool; history to its original turn, not current sensing. "
    "Null source/confidence means unavailable, not a verified sensor or certainty. "
    "timestamp_ms belongs to the source timeline (possibly offline video), not wall time; "
    "age_seconds is time since local receipt. "
    "UNKNOWN, MISSING, INVALID and STALE do not establish a current fact. "
    "WARMING_UP is insufficient observation; SUSPECTED is suspicion, not confirmed fatigue. "
    "Traffic level describes observed traffic density, not verified congestion. "
    "KNOWN only means the stored field passed current validation and age checks; "
    "it does not prove sensor correctness, environmental safety, or causal inference. "
    "Do not invent detection thresholds or clinical criteria absent from the supplied context. "
    "Do not infer driver impairment solely from PERCLOS unless a configured threshold is supplied. "
    "A detected lane does not establish that driving is safe. "
    "Music must not be described as eliminating fatigue."
)


def attach_field_evidence(
    context: dict[str, Any], manager: ContextManager
) -> dict[str, Any]:
    result = {}
    for topic, values in context.items():
        domain = topic if topic in {"driver", "road"} else "vehicle"
        aliases = (
            {
                "state": "navigation_state",
                "destination_id": "navigation_destination_id",
                "destination": "navigation_destination",
            }
            if topic == "navigation"
            else {}
        )
        evidence = {
            name: manager.field_evidence(domain, aliases.get(name, name))
            for name in values
            if name != "quality_status"
        }
        result[topic] = {**values, "field_evidence": evidence}
    return result

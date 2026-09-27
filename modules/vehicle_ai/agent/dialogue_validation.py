"""Fixed proposal schema and cross-field validation for dialogue turns."""

import math

from modules.vehicle_ai.agent.dialogue_source import (
    ambiguous_control,
    normalize_source_view,
    ordinal_from_source,
    parse_distance_source,
    parse_removal_source,
)
from modules.vehicle_ai.agent.dialogue_state import DialogueIntent


PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "changes", "reference", "evidence", "clarification_reason"],
    "properties": {
        "intent": {"type": "string", "enum": [item.value for item in DialogueIntent]},
        "changes": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "op", "value", "evidence"],
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["max_distance_km", "preferred_area", "unsupported"],
                    },
                    "op": {"type": "string", "enum": ["SET", "REMOVE"]},
                    "value": {"type": ["number", "string", "null"]},
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 240},
                },
            },
        },
        "reference": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["index", "name", "evidence"],
            "properties": {
                "index": {"type": ["integer", "null"], "minimum": 1},
                "name": {"type": ["string", "null"], "minLength": 1, "maxLength": 120},
                "evidence": {"type": "string", "minLength": 1, "maxLength": 240},
            },
        },
        "evidence": {"type": "string", "maxLength": 500},
        "clarification_reason": {"type": ["string", "null"], "maxLength": 240},
    },
}


def _matches_type(value, declared) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        if name == "object" and type(value) is dict:
            return True
        if name == "array" and type(value) is list:
            return True
        if name == "string" and type(value) is str:
            return True
        if name == "integer" and type(value) is int:
            return True
        if name == "number" and type(value) in {int, float}:
            return True
        if name == "null" and value is None:
            return True
    return False


def _schema_valid(value, schema) -> bool:
    if not _matches_type(value, schema["type"]):
        return False
    if value is None:
        return True
    if type(value) is float and not math.isfinite(value):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if type(value) is dict:
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= value.keys():
            return False
        if (
            schema.get("additionalProperties") is False
            and value.keys() - properties.keys()
        ):
            return False
        return all(
            key in properties and _schema_valid(item, properties[key])
            for key, item in value.items()
        )
    if type(value) is list:
        return len(value) <= schema.get("maxItems", math.inf) and all(
            _schema_valid(item, schema["items"]) for item in value
        )
    if type(value) is str:
        return len(value) >= schema.get("minLength", 0) and len(value) <= schema.get(
            "maxLength", math.inf
        )
    if type(value) in {int, float} and "minimum" in schema:
        return value >= schema["minimum"]
    return True


def _field_semantics(change: dict, text: str, changes: list[dict]) -> str | None:
    field, op, value, evidence = (
        change["field"],
        change["op"],
        change["value"],
        change["evidence"],
    )
    if op == "REMOVE":
        removal = parse_removal_source(field, value, text)
        if removal is None:
            return "SOURCE_OPERATION_MISMATCH"
        if removal.keeps_preferred_area and any(
            item["field"] == "preferred_area" and item["op"] == "SET"
            for item in changes
        ):
            return "SOURCE_OPERATION_MISMATCH"
        if removal.continuation_distance_km is not None and not any(
            item["field"] == "max_distance_km"
            and item["op"] == "SET"
            and type(item["value"]) in {int, float}
            and math.isclose(item["value"], removal.continuation_distance_km)
            for item in changes
        ):
            return "SOURCE_OPERATION_MISMATCH"
        return None
    if field == "max_distance_km":
        return None if type(value) in {int, float} else "INVALID_PROPOSAL"
    if field in {"preferred_area", "unsupported"}:
        valid = type(value) is str and bool(value) and value in evidence
        if field == "unsupported":
            valid = valid and value != "*"
        return None if valid else "SOURCE_VALUE_MISMATCH"
    return "INVALID_PROPOSAL"


def _value_semantics(change: dict, text: str) -> str | None:
    if change["op"] == "REMOVE" or change["field"] != "max_distance_km":
        return None
    value = change["value"]
    if (type(value) is float and not math.isfinite(value)) or value <= 0:
        return "INVALID_DISTANCE"
    source = parse_distance_source(text)
    if source.error is not None:
        return source.error
    if source.evidence not in normalize_source_view(change["evidence"]):
        return "SOURCE_EVIDENCE_MISMATCH"
    return (
        None
        if math.isclose(value, source.value_km, rel_tol=1e-9, abs_tol=1e-9)
        else "SOURCE_VALUE_MISMATCH"
    )


def validate_proposal(proposal: dict, text: str) -> str | None:
    if type(proposal) is not dict or not _schema_valid(proposal, PROPOSAL_SCHEMA):
        return "INVALID_PROPOSAL"
    intent, changes, reference = (
        proposal["intent"],
        proposal["changes"],
        proposal["reference"],
    )
    if intent in {"START", "UPDATE_CONSTRAINTS"} and reference is not None:
        return "CONFLICTING_INTENTS"
    if intent in {"CANCEL", "RESUME", "SIDE_QUESTION", "UNCLEAR"} and (
        reference is not None or changes
    ):
        return "CONFLICTING_INTENTS"
    if intent in {"SELECT", "ASK_CANDIDATE"}:
        if changes:
            return "CONFLICTING_INTENTS"
        if reference is None or (reference["index"] is None) == (
            reference["name"] is None
        ):
            return "INVALID_REFERENCE"
    elif reference is not None:
        return "CONFLICTING_INTENTS"
    reason = proposal["clarification_reason"]
    if (intent == "UNCLEAR") != (type(reason) is str and bool(reason.strip())):
        return "INTENT_REASON_MISMATCH"
    if intent == "START" and not proposal["evidence"]:
        return "INTENT_SOURCE_MISSING"

    evidence_items = [proposal["evidence"]]
    evidence_items.extend(change["evidence"] for change in changes)
    if reference is not None:
        evidence_items.append(reference["evidence"])
    if any(evidence and evidence not in text for evidence in evidence_items):
        return "SOURCE_EVIDENCE_MISMATCH"

    valid_remove_distance_continuation = False
    for change in changes:
        error = _field_semantics(change, text, changes)
        if error:
            return error
        if change["op"] == "REMOVE":
            removal = parse_removal_source(change["field"], change["value"], text)
            valid_remove_distance_continuation |= (
                removal is not None and removal.continuation_distance_km is not None
            )
    if intent in {"SELECT", "ASK_CANDIDATE"}:
        if reference["name"] is not None and reference["name"] not in text:
            return "SOURCE_REFERENCE_MISMATCH"
        if reference["index"] is not None and reference["index"] != ordinal_from_source(
            text, intent
        ):
            return "SOURCE_REFERENCE_MISMATCH"
    for change in changes:
        error = _value_semantics(change, text)
        if error:
            return error
    if (
        intent != "UNCLEAR"
        and ambiguous_control(text)
        and not (intent == "UPDATE_CONSTRAINTS" and valid_remove_distance_continuation)
    ):
        return "AMBIGUOUS_CONTROL"
    return None

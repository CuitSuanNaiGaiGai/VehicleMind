from __future__ import annotations

import json
import math

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import (
    ExpectedOutcome,
    ReplayObservation,
    ReplayScenario,
    ReplayStep,
    ScriptedResponse,
    freeze_mapping,
)


ROOT_KEYS = {
    "schema_version",
    "scenario_id",
    "title",
    "description",
    "media",
    "agent_responses",
    "steps",
    "expected",
}
OBSERVATION_KEYS = {"source", "confidence", "valid", "values"}
STEP_KEYS = {
    "at_ms",
    "cabin",
    "road",
    "vehicle",
    "user_text",
    "confirm_pending",
}


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _only_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        key = sorted(str(item) for item in unknown)[0]
        path = key if field == "scenario" else f"{field}.{key}"
        raise ValueError(f"unknown scenario field: {path}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return tuple(_text(item, f"{field}[{index}]") for index, item in enumerate(value))


def _observation(value: Any, field: str) -> ReplayObservation:
    document = _mapping(value, field)
    _only_keys(document, OBSERVATION_KEYS, field)
    missing = OBSERVATION_KEYS - set(document)
    if missing:
        raise ValueError(f"{field}.{sorted(missing)[0]} is required")

    confidence = document["confidence"]
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise ValueError(f"{field}.confidence must be null or a real number")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"{field}.confidence must be between zero and one")

    valid = document["valid"]
    if not isinstance(valid, bool):
        raise ValueError(f"{field}.valid must be a boolean")

    values = _mapping(document["values"], f"{field}.values")
    return ReplayObservation(
        source=_text(document["source"], f"{field}.source"),
        confidence=confidence,
        valid=valid,
        values=freeze_mapping(values),
    )


def _tool_call(value: Any, field: str) -> LLMToolCall:
    document = _mapping(value, field)
    _only_keys(document, {"id", "name", "arguments"}, field)
    arguments = _mapping(document.get("arguments"), f"{field}.arguments")
    return LLMToolCall(
        id=_text(document.get("id"), f"{field}.id"),
        name=_text(document.get("name"), f"{field}.name"),
        arguments=dict(arguments),
        arguments_json=json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    )


def _response(value: Any, index: int) -> ScriptedResponse:
    field = f"agent_responses[{index}]"
    document = _mapping(value, field)
    _only_keys(document, {"content", "tool_calls"}, field)
    content = document.get("content")
    if content is not None:
        content = _text(content, f"{field}.content")
    raw_calls = document.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise ValueError(f"{field}.tool_calls must be a list")
    calls = tuple(
        _tool_call(item, f"{field}.tool_calls[{call_index}]")
        for call_index, item in enumerate(raw_calls)
    )
    if content is None and not calls:
        raise ValueError(f"{field} must contain content or tool_calls")
    return ScriptedResponse(content=content, tool_calls=calls)


def _step(value: Any, index: int) -> ReplayStep:
    field = f"steps[{index}]"
    document = _mapping(value, field)
    _only_keys(document, STEP_KEYS, field)
    at_ms = document.get("at_ms")
    if isinstance(at_ms, bool) or not isinstance(at_ms, int) or at_ms < 0:
        raise ValueError(f"{field}.at_ms must be a non-negative integer")
    confirm = document.get("confirm_pending", False)
    if not isinstance(confirm, bool):
        raise ValueError(f"{field}.confirm_pending must be a boolean")
    user_text = document.get("user_text")
    if user_text is not None:
        user_text = _text(user_text, f"{field}.user_text")
    domains = {
        name: _observation(document[name], f"{field}.{name}")
        if name in document
        else None
        for name in ("cabin", "road", "vehicle")
    }
    if not any((*domains.values(), user_text, confirm)):
        raise ValueError(f"{field} must contain an observation or interaction")
    return ReplayStep(
        at_ms=at_ms,
        cabin=domains["cabin"],
        road=domains["road"],
        vehicle=domains["vehicle"],
        user_text=user_text,
        confirm_pending=confirm,
    )


def _media(value: Any, repository_root: Path) -> Mapping[str, Path]:
    document = _mapping(value, "media")
    _only_keys(document, {"cabin", "road"}, "media")
    if set(document) != {"cabin", "road"}:
        raise ValueError("media must contain cabin and road")
    root = repository_root.resolve()
    resolved: dict[str, Path] = {}
    for name in ("cabin", "road"):
        raw = _text(document[name], f"media.{name}")
        path = Path(raw)
        candidate = (root / path).resolve()
        if path.is_absolute() or not candidate.is_relative_to(root):
            raise ValueError(f"media.{name} must be a repository-relative path")
        if not candidate.is_file():
            raise ValueError(f"media.{name} does not exist: {raw}")
        resolved[name] = candidate
    return MappingProxyType(resolved)


def _expected(value: Any) -> ExpectedOutcome:
    document = _mapping(value, "expected")
    allowed = {
        "event_types",
        "successful_tools",
        "final_vehicle",
        "unauthorized_sensitive_executions",
    }
    _only_keys(document, allowed, "expected")
    missing = allowed - set(document)
    if missing:
        raise ValueError(f"expected.{sorted(missing)[0]} is required")
    unauthorized = document["unauthorized_sensitive_executions"]
    if (
        isinstance(unauthorized, bool)
        or not isinstance(unauthorized, int)
        or unauthorized < 0
    ):
        raise ValueError(
            "expected.unauthorized_sensitive_executions must be a non-negative integer"
        )
    return ExpectedOutcome(
        event_types=_string_tuple(document["event_types"], "expected.event_types"),
        successful_tools=_string_tuple(
            document["successful_tools"], "expected.successful_tools"
        ),
        final_vehicle=freeze_mapping(
            _mapping(document["final_vehicle"], "expected.final_vehicle")
        ),
        unauthorized_sensitive_executions=unauthorized,
    )


def load_replay_scenario(
    path: str | Path,
    *,
    repository_root: Path,
) -> ReplayScenario:
    scenario_path = Path(path)
    try:
        raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid scenario YAML: {exc}") from exc
    document = _mapping(raw, "scenario")
    _only_keys(document, ROOT_KEYS, "scenario")
    missing = ROOT_KEYS - set(document)
    if missing:
        raise ValueError(f"{sorted(missing)[0]} is required")
    if document["schema_version"] != 1:
        raise ValueError("schema_version must be 1")

    raw_responses = document["agent_responses"]
    if not isinstance(raw_responses, list) or not raw_responses:
        raise ValueError("agent_responses must be a non-empty list")
    responses = tuple(
        _response(value, index) for index, value in enumerate(raw_responses)
    )

    raw_steps = document["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("steps must be a non-empty list")
    steps = tuple(_step(value, index) for index, value in enumerate(raw_steps))
    previous = -1
    seen_user = False
    for index, step in enumerate(steps):
        if step.at_ms <= previous:
            raise ValueError(f"steps[{index}].at_ms must be strictly increasing")
        if step.confirm_pending and not seen_user:
            raise ValueError(
                f"steps[{index}].confirm_pending requires an earlier user turn"
            )
        if step.user_text is not None:
            seen_user = True
        previous = step.at_ms

    return ReplayScenario(
        schema_version=1,
        scenario_id=_text(document["scenario_id"], "scenario_id"),
        title=_text(document["title"], "title"),
        description=_text(document["description"], "description"),
        media=_media(document["media"], repository_root),
        responses=responses,
        steps=steps,
        expected=_expected(document["expected"]),
    )

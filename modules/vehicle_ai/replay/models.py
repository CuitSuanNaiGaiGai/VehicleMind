from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from collections.abc import Mapping

from modules.vehicle_ai.llm.base import LLMToolCall


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively immutable, detached mapping."""

    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in value.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class ScriptedResponse:
    content: str | None
    tool_calls: tuple[LLMToolCall, ...] = ()


@dataclass(frozen=True)
class ReplayObservation:
    source: str
    confidence: float | None
    valid: bool
    values: Mapping[str, object]


@dataclass(frozen=True)
class ReplayStep:
    at_ms: int
    cabin: ReplayObservation | None = None
    road: ReplayObservation | None = None
    vehicle: ReplayObservation | None = None
    user_text: str | None = None
    confirm_pending: bool = False


@dataclass(frozen=True)
class ExpectedOutcome:
    event_types: tuple[str, ...]
    successful_tools: tuple[str, ...]
    final_vehicle: Mapping[str, object]
    unauthorized_sensitive_executions: int = 0


@dataclass(frozen=True)
class ReplayScenario:
    schema_version: int
    scenario_id: str
    title: str
    description: str
    media: Mapping[str, Path]
    responses: tuple[ScriptedResponse, ...]
    steps: tuple[ReplayStep, ...]
    expected: ExpectedOutcome

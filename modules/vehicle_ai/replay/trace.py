from __future__ import annotations

import hashlib
import json

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from modules.vehicle_ai.context.contract import CONTEXT_SCHEMA_VERSION
from modules.vehicle_ai.replay.models import freeze_mapping


TRACE_KINDS = {
    "context_update",
    "event",
    "user_utterance",
    "agent_response",
    "pending_action",
    "confirmation",
    "tool_result",
    "assertion",
}


def plain_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain_value(item) for item in value]
    return value


@dataclass(frozen=True)
class TraceRecord:
    sequence: int
    at_ms: int
    kind: str
    data: Mapping[str, object]


@dataclass(frozen=True)
class AssertionResult:
    name: str
    passed: bool
    expected: object
    actual: object


@dataclass(frozen=True)
class ReplayResult:
    context_schema_version: int
    scenario_id: str
    passed: bool
    trace: tuple[TraceRecord, ...]
    semantic_sha256: str
    assertions: tuple[AssertionResult, ...]
    final_context: Mapping[str, object]
    metrics: Mapping[str, float]
    event_types: tuple[str, ...]
    successful_tools: tuple[str, ...]
    unauthorized_sensitive_executions: int
    remaining_scripted_responses: int


class TraceRecorder:
    def __init__(self) -> None:
        self._records: list[TraceRecord] = []

    def add(self, *, at_ms: int, kind: str, data: Mapping[str, object]) -> None:
        if kind not in TRACE_KINDS:
            raise ValueError(f"unknown trace kind: {kind}")
        self._records.append(
            TraceRecord(
                sequence=len(self._records),
                at_ms=at_ms,
                kind=kind,
                data=freeze_mapping(data),
            )
        )

    def records(self) -> tuple[TraceRecord, ...]:
        return tuple(self._records)


def semantic_digest(
    scenario_id: str,
    records: tuple[TraceRecord, ...],
) -> str:
    document = {
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "trace": [
            {
                "sequence": item.sequence,
                "at_ms": item.at_ms,
                "kind": item.kind,
                "data": plain_value(item.data),
            }
            for item in records
        ],
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_plain_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(plain_value(value))

"""Cooperative turn deadlines; never start another operation after expiry."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentBudgetConfig:
    turn_timeout_seconds: float = 90.0
    max_tool_calls: int = 10
    max_tool_rounds: int = 5
    max_task_trace_events: int = 200

    @classmethod
    def load(cls, path: Path = Path("modules/config/agent.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        required = {
            "turn_timeout_seconds",
            "max_tool_calls",
            "max_tool_rounds",
            "max_task_trace_events",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("invalid agent budget config keys")
        config = cls(**raw)
        if (
            config.turn_timeout_seconds <= 0
            or min(
                config.max_tool_calls,
                config.max_tool_rounds,
                config.max_task_trace_events,
            )
            < 1
        ):
            raise ValueError("agent budget limits must be positive")
        if type(config.turn_timeout_seconds) not in {int, float} or any(
            type(value) is not int
            for value in (
                config.max_tool_calls,
                config.max_tool_rounds,
                config.max_task_trace_events,
            )
        ):
            raise ValueError("agent budget types are invalid")
        return config


@dataclass
class TurnBudget:
    seconds: float
    max_calls: int
    clock: Callable[[], float]
    deadline: float = field(init=False)
    calls: int = 0
    signatures: set[str] = field(default_factory=set)
    used_calls: set[str] = field(default_factory=set)

    def __post_init__(self):
        if not math.isfinite(self.seconds) or self.seconds <= 0 or self.max_calls < 1:
            raise ValueError("turn timeout and tool call budget must be positive")
        self.deadline = self.clock() + self.seconds

    def remaining(self) -> float:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise BudgetExceeded("TIME_BUDGET")
        return remaining

    def claim(self, name: str, arguments: dict) -> None:
        self.remaining()
        if self.calls >= self.max_calls:
            raise BudgetExceeded("TOOL_BUDGET")
        signature = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
        if signature in self.signatures:
            raise BudgetExceeded("REPEATED_CALL")
        self.signatures.add(signature)
        self.used_calls.add(signature)
        self.calls += 1

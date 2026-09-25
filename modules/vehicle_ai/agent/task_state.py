"""Observable task lifecycle, independent of natural-language model replies."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class TaskStatus(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    AWAITING_INPUT = "AWAITING_INPUT"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: uuid4().hex)
    goal: str = ""
    status: TaskStatus = TaskStatus.IDLE
    reason: str | None = None
    last_tool_result: dict[str, Any] | None = None
    reconciliation: dict[str, Any] | None = None
    pending_action: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, status: TaskStatus, reason: str | None = None) -> None:
        self.transitions.append({"from": self.status, "to": status, "reason": reason})
        self.status, self.reason = status, reason

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))

    def finish(self, pending: bool) -> None:
        if self.status in {TaskStatus.FAILED, TaskStatus.AWAITING_INPUT}:
            return
        if pending:
            self.transition(TaskStatus.AWAITING_CONFIRMATION)
        elif any(
            not result["success"] and result.get("error") != "CONFIRMATION_REQUIRED"
            for result in self.tool_results
        ):
            self.transition(
                TaskStatus.FAILED,
                next(
                    result.get("error") or "TOOL_ERROR"
                    for result in self.tool_results
                    if not result["success"]
                    and result.get("error") != "CONFIRMATION_REQUIRED"
                ),
            )
        else:
            self.transition(TaskStatus.COMPLETED)

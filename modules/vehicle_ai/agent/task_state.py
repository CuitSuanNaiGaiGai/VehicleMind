"""Observable task lifecycle, independent of natural-language model replies."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class TaskStatus(StrEnum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
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
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, status: TaskStatus, reason: str | None = None) -> None:
        self.transitions.append({"from": self.status, "to": status, "reason": reason})
        self.status, self.reason = status, reason

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))

    def finish(self, pending: bool) -> None:
        if self.status is TaskStatus.FAILED:
            return
        if pending:
            self.transition(TaskStatus.AWAITING_CONFIRMATION)
        elif self.last_tool_result and not self.last_tool_result["success"]:
            self.transition(
                TaskStatus.FAILED, self.last_tool_result.get("error") or "TOOL_ERROR"
            )
        else:
            self.transition(TaskStatus.COMPLETED)

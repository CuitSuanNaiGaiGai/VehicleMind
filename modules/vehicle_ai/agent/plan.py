"""Finite, auditable task plans for bounded vehicle actions."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PlanStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    STOPPED_NO_RESULT = "STOPPED_NO_RESULT"


class PlanStepStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PlanTransitionError(RuntimeError):
    """Raised when a plan transition violates its bounded lifecycle."""


@dataclass
class PlanStep:
    name: str
    status: PlanStepStatus
    started_at: float
    finished_at: float | None
    success_condition: str
    evidence_summary: str = ""
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success_condition": self.success_condition,
            "evidence_summary": self.evidence_summary,
            "error_code": self.error_code,
        }


@dataclass
class TaskPlan:
    goal: str
    max_steps: int
    max_recoveries: int
    steps: list[PlanStep] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_poi_id: str | None = None
    recovery_count: int = 0
    recovery_reasons: list[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.RUNNING
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("plan goal must not be empty")
        if self.max_steps < 1 or self.max_recoveries < 0:
            raise ValueError("plan budgets must be non-negative and steps positive")

    @property
    def remaining_step_budget(self) -> int:
        return max(0, self.max_steps - len(self.steps))

    @property
    def remaining_recovery_budget(self) -> int:
        return max(0, self.max_recoveries - self.recovery_count)

    def start_step(
        self, name: str, *, success_condition: str, at: float | None = None
    ) -> PlanStep:
        self._require_running()
        if any(step.status is PlanStepStatus.RUNNING for step in self.steps):
            raise PlanTransitionError("STEP_ALREADY_RUNNING")
        if len(self.steps) >= self.max_steps:
            self.stop(PlanStatus.FAILED, "STEP_BUDGET_EXCEEDED")
            raise PlanTransitionError("STEP_BUDGET_EXCEEDED")
        step = PlanStep(
            name=name,
            status=PlanStepStatus.RUNNING,
            started_at=time.time() if at is None else at,
            finished_at=None,
            success_condition=success_condition,
        )
        self.steps.append(step)
        return step

    def finish_step(
        self,
        name: str,
        *,
        success: bool,
        evidence_summary: str = "",
        error_code: str | None = None,
        at: float | None = None,
    ) -> PlanStep:
        step = self._active_step(name)
        step.status = PlanStepStatus.SUCCEEDED if success else PlanStepStatus.FAILED
        step.finished_at = time.time() if at is None else at
        step.evidence_summary = evidence_summary[:500]
        step.error_code = error_code
        return step

    def skip_step(
        self,
        name: str,
        *,
        reason: str,
        success_condition: str,
        at: float | None = None,
    ) -> PlanStep:
        step = self.start_step(name, success_condition=success_condition, at=at)
        step.status = PlanStepStatus.SKIPPED
        step.finished_at = time.time() if at is None else at
        step.evidence_summary = reason[:500]
        return step

    def claim_recovery(self, reason: str) -> bool:
        self._require_running()
        if self.recovery_count >= self.max_recoveries:
            if self.steps and self.steps[-1].status is PlanStepStatus.RUNNING:
                step = self.steps[-1]
                step.status = PlanStepStatus.FAILED
                step.finished_at = time.time()
                step.evidence_summary = "计划恢复次数已达到配置上限。"
                step.error_code = "RECOVERY_BUDGET_EXCEEDED"
            self.stop(PlanStatus.FAILED, "RECOVERY_BUDGET_EXCEEDED")
            return False
        self.recovery_count += 1
        self.recovery_reasons.append(reason[:160])
        return True

    def complete(self, *, at: float | None = None) -> None:
        self._require_running()
        if any(step.status is PlanStepStatus.RUNNING for step in self.steps):
            raise PlanTransitionError("STEP_STILL_RUNNING")
        self.status = PlanStatus.COMPLETED
        self.terminal_reason = "SUCCESS"

    def stop(self, status: PlanStatus, reason: str) -> None:
        if status is PlanStatus.RUNNING or status is PlanStatus.COMPLETED:
            raise ValueError("stop requires a failed, cancelled, or no-result status")
        if self.status is not PlanStatus.RUNNING:
            return
        self.status = status
        self.terminal_reason = reason[:160]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status.value,
            "max_steps": self.max_steps,
            "max_recoveries": self.max_recoveries,
            "steps": [step.to_dict() for step in self.steps],
            "candidates": deepcopy(self.candidates),
            "selected_poi_id": self.selected_poi_id,
            "recovery_count": self.recovery_count,
            "recovery_reasons": list(self.recovery_reasons),
            "remaining_step_budget": self.remaining_step_budget,
            "remaining_recovery_budget": self.remaining_recovery_budget,
            "terminal_reason": self.terminal_reason,
        }

    def _active_step(self, name: str) -> PlanStep:
        self._require_running()
        if not self.steps or self.steps[-1].name != name:
            raise PlanTransitionError("STEP_NOT_RUNNING")
        step = self.steps[-1]
        if step.status is not PlanStepStatus.RUNNING:
            raise PlanTransitionError("STEP_NOT_RUNNING")
        return step

    def _require_running(self) -> None:
        if self.status is not PlanStatus.RUNNING:
            raise PlanTransitionError("PLAN_TERMINAL")

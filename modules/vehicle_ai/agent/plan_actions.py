"""Step tracing, navigation verification, and bounded recovery actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modules.vehicle_ai.agent.plan import (
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    PlanTransitionError,
    TaskPlan,
)
from modules.vehicle_ai.agent.session import record
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.tools.base import ToolResult

if TYPE_CHECKING:
    from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent


class PlanActionsMixin:
    """Common plan-step trace operations and navigation outcome handling."""

    def _start_step(
        self, agent: VehicleAgent, plan: TaskPlan, name: str, condition: str
    ) -> PlanStep:
        try:
            step = plan.start_step(name, success_condition=condition)
        except PlanTransitionError as error:
            if plan.status is PlanStatus.RUNNING:
                self._stop(
                    agent,
                    plan,
                    PlanStatus.FAILED,
                    str(error),
                    task_status=TaskStatus.FAILED,
                )
            else:
                self._emit(
                    agent, plan, plan.steps[-1] if plan.steps else None, "PLAN_STOPPED"
                )
                agent.task.transition(TaskStatus.FAILED, str(error))
            raise
        self._emit(agent, plan, step, "STEP_STARTED")
        return step

    def _finish_step(
        self,
        agent: VehicleAgent,
        plan: TaskPlan,
        step: PlanStep,
        *,
        success: bool,
        evidence: str,
        error: str | None = None,
    ) -> None:
        plan.finish_step(
            step.name,
            success=success,
            evidence_summary=evidence,
            error_code=error,
        )
        self._emit(agent, plan, step, "STEP_FINISHED")

    def _complete_step(
        self,
        agent: VehicleAgent,
        plan: TaskPlan,
        name: str,
        condition: str,
        evidence: str,
    ) -> PlanStep:
        step = self._start_step(agent, plan, name, condition)
        self._finish_step(agent, plan, step, success=True, evidence=evidence)
        return step

    def _sync_pending_action(self, agent: VehicleAgent, plan: TaskPlan) -> bool:
        pending = agent.pending_actions.get()
        if pending is None:
            self._stop(
                agent,
                plan,
                PlanStatus.FAILED,
                "PENDING_ACTION_MISSING",
                task_status=TaskStatus.FAILED,
            )
            return False
        agent.task.pending_action = pending.to_dict()
        return True

    def _stop(
        self,
        agent: VehicleAgent,
        plan: TaskPlan,
        status: PlanStatus,
        reason: str,
        *,
        task_status: TaskStatus,
    ) -> None:
        if plan.status is PlanStatus.RUNNING:
            active = next(
                (
                    step
                    for step in reversed(plan.steps)
                    if step.status is PlanStepStatus.RUNNING
                ),
                None,
            )
            if active is not None:
                self._finish_step(
                    agent,
                    plan,
                    active,
                    success=False,
                    evidence=f"计划停止：{reason}。",
                    error=reason,
                )
        plan.stop(status, reason)
        self._emit(agent, plan, plan.steps[-1] if plan.steps else None, "PLAN_STOPPED")
        agent.task.transition(task_status, reason)

    @staticmethod
    def _active_step(plan: TaskPlan, name: str) -> PlanStep | None:
        if plan.steps and plan.steps[-1].name == name:
            step = plan.steps[-1]
            return step if step.status is PlanStepStatus.RUNNING else None
        return None

    @staticmethod
    def _candidate(plan: TaskPlan, poi_id: str) -> dict[str, Any] | None:
        return next(
            (item for item in plan.candidates if item.get("poi_id") == poi_id), None
        )

    @staticmethod
    def display_name(candidate: dict[str, Any]) -> str:
        return str(
            candidate.get("display_name_zh") or candidate.get("name") or "休息地点"
        )

    @staticmethod
    def _emit(
        agent: VehicleAgent,
        plan: TaskPlan,
        step: PlanStep | None,
        event: str,
    ) -> None:
        record(
            agent,
            "plan_step",
            source="bounded_rest_plan",
            quality="KNOWN",
            event=event,
            step=step.to_dict() if step else None,
            plan=plan.to_dict(),
        )

    def _verify_navigation(self, agent: VehicleAgent, plan: TaskPlan) -> None:
        step = self._start_step(
            agent, plan, "VERIFY", "车况目的地与已确认 POI 一致且状态为 ACTIVE"
        )
        vehicle = agent.context_manager.get_context().vehicle
        matches = (
            vehicle.navigation_state.value == "ACTIVE"
            and vehicle.navigation_destination_id == plan.selected_poi_id
        )
        self._finish_step(
            agent,
            plan,
            step,
            success=matches,
            evidence=(
                "车况状态回读与已确认地点一致。"
                if matches
                else "车况状态回读未能确认目标地点。"
            ),
            error=None if matches else "NAVIGATION_STATE_MISMATCH",
        )
        if matches:
            plan.complete()
            self._emit(agent, plan, step, "PLAN_COMPLETED")
            agent.task.transition(TaskStatus.COMPLETED, "PLAN_VERIFIED")
        else:
            self._stop(
                agent,
                plan,
                PlanStatus.FAILED,
                "NAVIGATION_STATE_MISMATCH",
                task_status=TaskStatus.FAILED,
            )

    def _propose_alternative(
        self, agent: VehicleAgent, plan: TaskPlan, failure: ToolResult
    ) -> ToolResult:
        if not plan.claim_recovery("POI_UNAVAILABLE"):
            self._stop(
                agent,
                plan,
                PlanStatus.FAILED,
                "RECOVERY_BUDGET_EXCEEDED",
                task_status=TaskStatus.FAILED,
            )
            return failure
        alternative = next(
            (
                candidate
                for candidate in plan.candidates
                if candidate.get("poi_id") != plan.selected_poi_id
            ),
            None,
        )
        if alternative is None:
            self._stop(
                agent,
                plan,
                PlanStatus.STOPPED_NO_RESULT,
                "NO_ALTERNATIVE",
                task_status=TaskStatus.AWAITING_INPUT,
            )
            return failure
        plan.selected_poi_id = str(alternative["poi_id"])
        name = self.display_name(alternative)
        recovery_step = self._complete_step(
            agent,
            plan,
            "RECOVER",
            "首选失败后只提出一个不同的已搜索候选，不自动导航",
            f"首选地点不可用；提供替代候选 {name}，等待新的用户确认。",
        )
        agent.confirmations.stage(
            "start_navigation",
            {"poi_id": plan.selected_poi_id},
            user_intent=agent.current_user_intent,
            display_text=f"确认导航至{name}",
            metadata={
                "candidate": alternative,
                "recovery_reason": "POI_UNAVAILABLE",
            },
        )
        if not self._sync_pending_action(agent, plan):
            return ToolResult(
                False,
                "无法建立待确认导航动作，计划已安全停止。",
                error="PENDING_ACTION_MISSING",
            )
        self._start_step(
            agent,
            plan,
            "AWAIT_CONFIRMATION",
            "替代 POI 使用新 action ID 且必须重新确认",
        )
        agent.task.transition(TaskStatus.AWAITING_CONFIRMATION, "ALTERNATIVE_PENDING")
        self._emit(agent, plan, recovery_step, "ALTERNATIVE_PROPOSED")
        return ToolResult(
            False,
            f"首选地点不可用，已准备替代地点{name}；尚未导航，需再次确认。",
            error="ALTERNATIVE_PENDING",
            data={
                "failed_poi_id": failure.data.get("poi_id"),
                "alternative_poi_id": plan.selected_poi_id,
                "requires_new_confirmation": True,
                "simulated": True,
            },
        )

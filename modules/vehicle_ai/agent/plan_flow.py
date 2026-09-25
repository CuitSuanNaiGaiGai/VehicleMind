"""Bounded orchestration for rest-location search and navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modules.config.agent_plan import AgentPlanConfig
from modules.vehicle_ai.agent.plan import (
    PlanStatus,
    PlanStepStatus,
    TaskPlan,
)
from modules.vehicle_ai.agent.plan_actions import PlanActionsMixin
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.agent.pending_intent import search_result_matches_target
from modules.vehicle_ai.tools.base import ToolResult

if TYPE_CHECKING:
    from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent


REST_LOCATION_TERMS = (
    "服务区",
    "休息区",
    "停车场",
    "附近停车",
    "找地方休息",
    "休息地点",
)


class TaskPlanFlow(PlanActionsMixin):
    """Keep model-proposed rest-stop tool calls inside a finite state plan."""

    def __init__(self, config: AgentPlanConfig | None = None) -> None:
        self.config = config or AgentPlanConfig.load()

    @staticmethod
    def is_rest_location_request(text: str) -> bool:
        normalized = text.casefold()
        return any(term in normalized for term in REST_LOCATION_TERMS)

    def begin(self, agent: VehicleAgent, goal: str) -> TaskPlan | None:
        if not self.is_rest_location_request(goal):
            return None
        current = agent.task.plan
        if current is not None and current.status is PlanStatus.RUNNING:
            if current.goal == goal:
                return current
            self.cancel(agent, "TARGET_CHANGED")
            agent.task.transition(TaskStatus.RUNNING, "NEW_PLAN")
        plan = TaskPlan(
            goal=goal,
            max_steps=self.config.max_steps,
            max_recoveries=self.config.max_recoveries,
        )
        agent.task.plan = plan
        self._complete_step(
            agent,
            plan,
            "UNDERSTAND",
            "用户请求需要查询休息地点",
            "识别到休息地点任务；地点与导航写操作仍需候选核验和用户确认。",
        )
        return plan

    def before_tool(
        self,
        agent: VehicleAgent,
        name: str,
        arguments: dict[str, Any],
        *,
        target: str | None = None,
    ) -> str | None:
        plan = agent.task.plan
        if name == "search_nearby_rest_area" and plan is None:
            plan = self.begin(agent, agent.current_user_intent)
        if plan is None:
            return None
        if plan.status is not PlanStatus.RUNNING:
            if name in {"search_nearby_rest_area", "start_navigation"}:
                return "当前休息地点计划已结束，请用新的请求重新开始。"
            return None

        if name == "search_vehicle_knowledge":
            self._start_step(
                agent,
                plan,
                "KNOWLEDGE",
                "如需说明休息建议，返回匹配的知识来源或明确无命中",
            )
        elif name == "search_nearby_rest_area":
            self._start_step(
                agent,
                plan,
                "SEARCH",
                "返回最多配置数量的规范模拟地点候选，或明确无结果",
            )
        elif name == "start_navigation":
            poi_id = str(arguments.get("poi_id", ""))
            pending = agent.pending_actions.get()
            if (
                pending is not None
                and pending.tool_name == "start_navigation"
                and pending.arguments.get("poi_id") == poi_id
                and self._active_step(plan, "AWAIT_CONFIRMATION") is not None
            ):
                return None
            candidate = self._candidate(plan, poi_id)
            if candidate is None:
                reason = (
                    "TARGET_NOT_FOUND" if target is not None else "UNGROUNDED_POI_ID"
                )
                self._stop(
                    agent,
                    plan,
                    PlanStatus.FAILED,
                    reason,
                    task_status=(
                        TaskStatus.AWAITING_INPUT
                        if target is not None
                        else TaskStatus.FAILED
                    ),
                )
                return (
                    "未找到与新目标匹配的地点，未创建待确认导航。"
                    if target is not None
                    else "地点不在本轮模拟搜索候选中，已停止且未创建导航操作。"
                )
            plan.selected_poi_id = poi_id
            self._complete_step(
                agent,
                plan,
                "SELECT",
                "选择 ID 必须来自当前搜索候选",
                f"已选择候选：{self.display_name(candidate)}。",
            )
            self._start_step(
                agent,
                plan,
                "AWAIT_CONFIRMATION",
                "为该规范 POI 建立一次性待确认导航动作；此时尚未执行",
            )
        return None

    def observe_tool(
        self,
        agent: VehicleAgent,
        name: str,
        arguments: dict[str, Any],
        result: ToolResult,
        target: str | None = None,
        allow_retry: bool = False,
    ) -> None:
        plan = agent.task.plan
        if plan is None or plan.status is not PlanStatus.RUNNING:
            return
        if name == "search_vehicle_knowledge":
            step = self._active_step(plan, "KNOWLEDGE")
            if step is not None:
                self._finish_step(
                    agent,
                    plan,
                    step,
                    success=result.success,
                    evidence=(
                        "返回带来源的知识证据。"
                        if result.success
                        else f"知识检索未命中或失败：{result.error or 'UNKNOWN'}。"
                    ),
                    error=result.error,
                )
        elif name == "search_nearby_rest_area":
            step = self._active_step(plan, "SEARCH")
            if step is None:
                return
            raw_candidates = result.data.get("candidates", [])
            candidates = [
                dict(item)
                for item in raw_candidates
                if isinstance(item, dict)
                and isinstance(item.get("poi_id"), str)
                and item.get("poi_id")
            ][: self.config.max_candidates]
            plan.candidates = candidates
            self._finish_step(
                agent,
                plan,
                step,
                success=result.success and bool(candidates),
                evidence=(
                    f"收到 {len(candidates)} 个模拟地点候选。"
                    if candidates
                    else "搜索没有返回可用候选。"
                ),
                error=result.error,
            )
            if not result.success or not candidates:
                terminal_reason = result.error or "NO_RESULTS"
                if result.data.get("retryable") and allow_retry:
                    self._start_step(
                        agent,
                        plan,
                        "SEARCH",
                        "对可重试的只读查询最多再执行一次",
                    )
                    return
                status = (
                    PlanStatus.STOPPED_NO_RESULT
                    if terminal_reason == "NO_RESULTS"
                    else PlanStatus.FAILED
                )
                self._stop(
                    agent,
                    plan,
                    status,
                    terminal_reason,
                    task_status=TaskStatus.AWAITING_INPUT,
                )
            elif target is not None:
                selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if search_result_matches_target(target, candidate)
                    ),
                    None,
                )
                if selected is None:
                    self._stop(
                        agent,
                        plan,
                        PlanStatus.FAILED,
                        "TARGET_NOT_FOUND",
                        task_status=TaskStatus.AWAITING_INPUT,
                    )
                else:
                    self._stage_candidate(
                        agent,
                        plan,
                        selected,
                        reason="已匹配用户明确指定的规范地点。",
                    )

    def begin_confirmation(
        self, agent: VehicleAgent, action: dict[str, Any] | None
    ) -> bool:
        plan = agent.task.plan
        if (
            plan is None
            or plan.status is not PlanStatus.RUNNING
            or not action
            or action.get("tool_name") != "start_navigation"
        ):
            return False
        poi_id = action.get("arguments", {}).get("poi_id")
        if (
            poi_id != plan.selected_poi_id
            or self._active_step(plan, "AWAIT_CONFIRMATION") is None
        ):
            return False
        step = self._active_step(plan, "AWAIT_CONFIRMATION")
        assert step is not None
        self._finish_step(
            agent,
            plan,
            step,
            success=True,
            evidence="用户通过显式确认入口提交本次动作授权。",
        )
        self._start_step(
            agent,
            plan,
            "EXECUTE",
            "只执行本 PendingAction 绑定的规范 POI；不自动重放写请求",
        )
        return True

    def finish_confirmation(
        self,
        agent: VehicleAgent,
        action: dict[str, Any] | None,
        result: ToolResult,
    ) -> ToolResult | None:
        plan = agent.task.plan
        if (
            plan is None
            or plan.status is not PlanStatus.RUNNING
            or not action
            or action.get("tool_name") != "start_navigation"
        ):
            return None
        execute_step = self._active_step(plan, "EXECUTE")
        if execute_step is None:
            return None
        success = result.success
        self._finish_step(
            agent,
            plan,
            execute_step,
            success=success,
            evidence=result.message[:240],
            error=result.error,
        )
        if result.success:
            self._verify_navigation(agent, plan)
            return result
        if result.data.get("outcome_unknown"):
            self._stop(
                agent,
                plan,
                PlanStatus.FAILED,
                "WRITE_OUTCOME_UNKNOWN",
                task_status=TaskStatus.AWAITING_INPUT,
            )
            return result
        if result.error == "POI_UNAVAILABLE":
            return self._propose_alternative(agent, plan, result)
        self._stop(
            agent,
            plan,
            PlanStatus.FAILED,
            result.error or "NAVIGATION_FAILED",
            task_status=TaskStatus.FAILED,
        )
        return result

    def cancel(
        self,
        agent: VehicleAgent,
        reason: str,
        *,
        task_status: TaskStatus = TaskStatus.CANCELLED,
    ) -> bool:
        plan = agent.task.plan
        if plan is None or plan.status is not PlanStatus.RUNNING:
            return False
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
                evidence="计划在当前步骤被用户取消或目标变更。",
                error=reason,
            )
        self._stop(
            agent,
            plan,
            PlanStatus.CANCELLED,
            reason,
            task_status=task_status,
        )
        return True

    def finish_turn(self, agent: VehicleAgent, answer: str) -> None:
        plan = agent.task.plan
        intent = agent.current_user_intent.casefold()
        navigation_declined = any(
            phrase in intent
            for phrase in (
                "不要导航",
                "不要直接启动导航",
                "暂不导航",
                "不用导航",
                "不要开导航",
                "do not navigate",
                "don't navigate",
                "no navigation",
            )
        )
        if (
            plan is None
            or plan.status is not PlanStatus.RUNNING
            or agent.pending_actions.get() is not None
            or not plan.candidates
            or navigation_declined
            or not any(
                token in answer.casefold()
                for token in ("导航", "确认", "带你去", "navigate", "confirm")
            )
        ):
            return
        candidate = min(
            plan.candidates, key=lambda item: item.get("distance_km", float("inf"))
        )
        self._stage_candidate(
            agent,
            plan,
            candidate,
            reason="模型建议将最近的已检索地点作为候选；仅生成待确认动作。",
        )

    def stop_current(self, agent: VehicleAgent, reason: str) -> None:
        plan = agent.task.plan
        if plan is not None and plan.status is PlanStatus.RUNNING:
            self._stop(
                agent,
                plan,
                PlanStatus.FAILED,
                reason,
                task_status=TaskStatus.FAILED,
            )

    def claim_read_retry(
        self, agent: VehicleAgent, name: str, result: ToolResult
    ) -> bool:
        plan = agent.task.plan
        if plan is None:
            return True
        if plan.status is not PlanStatus.RUNNING:
            return False
        allowed = plan.claim_recovery(
            f"READ_ONLY_RETRY:{name}:{result.error or 'TRANSIENT_ERROR'}"
        )
        if not allowed:
            self._stop(
                agent,
                plan,
                PlanStatus.FAILED,
                "RECOVERY_BUDGET_EXCEEDED",
                task_status=TaskStatus.AWAITING_INPUT,
            )
        return allowed

    def candidate_for(self, agent: VehicleAgent, poi_id: str) -> dict[str, Any] | None:
        plan = agent.task.plan
        return self._candidate(plan, poi_id) if plan is not None else None

    def _stage_candidate(
        self,
        agent: VehicleAgent,
        plan: TaskPlan,
        candidate: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        plan.selected_poi_id = str(candidate["poi_id"])
        name = self.display_name(candidate)
        self._complete_step(
            agent,
            plan,
            "SELECT",
            "选中 ID 必须来自当前搜索返回的规范候选",
            f"{reason}候选地点：{name}。",
        )
        self._start_step(
            agent,
            plan,
            "AWAIT_CONFIRMATION",
            "只准备用户明确指定的规范 POI，等待本次动作确认",
        )
        agent.confirmations.stage(
            "start_navigation",
            {"poi_id": plan.selected_poi_id},
            user_intent=agent.current_user_intent,
            display_text=f"确认导航至{name}",
            metadata={"candidate": candidate, "selection_reason": reason},
        )
        if not self._sync_pending_action(agent, plan):
            return
        agent.task.transition(TaskStatus.AWAITING_CONFIRMATION, "TARGET_MATCHED")

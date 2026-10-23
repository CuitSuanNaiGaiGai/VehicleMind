"""Version and expiry checks for rest-location candidates in dialogue."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from modules.vehicle_ai.agent.budget import BudgetExceeded, TurnBudget
from modules.vehicle_ai.agent.dialogue_state import CandidateSnapshot
from modules.vehicle_ai.agent.plan import PlanStepStatus
from modules.vehicle_ai.agent.target_resolution import resolve_target
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.tools.base import ToolResult

if TYPE_CHECKING:
    from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent


SEARCH_TOOL = "search_nearby_rest_area"


def candidate_is_current(agent: VehicleAgent) -> bool:
    task, plan = agent.task, agent.task.plan
    snapshot = plan.candidate_snapshot if plan is not None else None
    if (
        task.status
        in {
            TaskStatus.IDLE,
            TaskStatus.CANCELLED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        }
        or plan is None
        or plan.status.value != "RUNNING"
        or snapshot is None
        or snapshot.constraint_revision != task.revision
        or agent.pending_actions.now() >= snapshot.expires_at
        or snapshot.candidate_set_id != task.last_presented_candidate_set_id
        or snapshot.presented_turn_id is None
        or not snapshot.presented_poi_ids
    ):
        return False
    counts = Counter(
        str(item.get("poi_id")) for item in plan.candidates if item.get("poi_id")
    )
    return all(counts.get(poi_id) == 1 for poi_id in snapshot.presented_poi_ids)


def resolve_reference(
    agent: VehicleAgent, reference: dict | None
) -> dict[str, Any] | None:
    if not candidate_is_current(agent) or type(reference) is not dict:
        return None
    plan = agent.task.plan
    snapshot = plan.candidate_snapshot
    if type(reference.get("index")) is int and reference["index"] > 0:
        index = reference["index"] - 1
        if index >= len(snapshot.presented_poi_ids):
            return None
        poi_id = snapshot.presented_poi_ids[index]
        matches = [item for item in plan.candidates if item.get("poi_id") == poi_id]
        return dict(matches[0]) if len(matches) == 1 else None
    name = reference.get("name")
    if type(name) is not str or not name.strip():
        return None
    result = resolve_target(name, plan.candidates)
    return dict(result.candidate) if result.status == "matched" else None


def _clear(agent: VehicleAgent, reason: str, *, awaiting_input: bool = True) -> None:
    plan = agent.task.plan
    pending = agent.pending_actions.get()
    if pending is not None and pending.tool_name == "start_navigation":
        agent.pending_actions.clear()
    agent.task.pending_action = None
    if plan is not None:
        plan.candidates.clear()
        plan.selected_poi_id = None
        plan.candidate_snapshot = None
    agent.task.last_presented_candidate_set_id = None
    if plan is not None and plan.status.value == "RUNNING":
        active = next(
            (
                step
                for step in reversed(plan.steps)
                if step.status is PlanStepStatus.RUNNING
            ),
            None,
        )
        if active is not None and active.name == "AWAIT_CONFIRMATION":
            agent.plan_flow._finish_step(
                agent, plan, active, success=False, evidence=reason, error=reason
            )
            if awaiting_input:
                agent.task.transition(TaskStatus.AWAITING_INPUT, reason)


def invalidate_candidates(agent: VehicleAgent, reason: str) -> None:
    _clear(agent, reason)


def _search_arguments(agent: VehicleAgent) -> dict[str, Any]:
    supported = {"max_distance_km", "preferred_area"}
    return {
        name: value.value
        for name, value in agent.task.constraints.items()
        if name in supported
    }


def _record_search(agent: VehicleAgent, arguments: dict, result: ToolResult) -> None:
    agent.task.last_tool_result = result.to_dict()
    agent.task.tool_results.append(result.to_dict())
    from modules.vehicle_ai.agent.session import record

    record(
        agent,
        "tool_result",
        source="dialogue_candidates",
        quality="TOOL_RESULT",
        tool_name=SEARCH_TOOL,
        arguments=arguments,
        result=result.to_dict(),
    )


def search_candidates(agent: VehicleAgent, budget: TurnBudget) -> str:
    plan = agent.task.plan
    if plan is None:
        return "当前没有可执行的休息地点计划，请重新说明查找目标。"
    _clear(agent, "CANDIDATE_REFRESH")
    arguments = _search_arguments(agent)
    try:
        budget.claim(SEARCH_TOOL, arguments)
    except BudgetExceeded as error:
        return f"本轮搜索预算已用完（{error}），请稍后重新查询。"
    blocked = agent.plan_flow.before_tool(agent, SEARCH_TOOL, arguments)
    if blocked:
        return blocked
    result = agent.tool_registry.execute(
        SEARCH_TOOL, arguments, user_intent=agent.current_user_intent
    )
    retry = bool(result.data.get("retryable") and plan.remaining_recovery_budget > 0)
    if retry:
        try:
            budget.claim_retry(SEARCH_TOOL, arguments)
        except BudgetExceeded:
            retry = False
    if retry and agent.plan_flow.claim_read_retry(agent, SEARCH_TOOL, result):
        agent.plan_flow.observe_tool(
            agent, SEARCH_TOOL, arguments, result, allow_retry=True
        )
        _record_search(agent, arguments, result)
        result = agent.tool_registry.execute(
            SEARCH_TOOL, arguments, user_intent=agent.current_user_intent
        )
        agent.plan_flow.observe_tool(
            agent, SEARCH_TOOL, arguments, result, allow_retry=False
        )
    else:
        agent.plan_flow.observe_tool(
            agent, SEARCH_TOOL, arguments, result, allow_retry=False
        )
    _record_search(agent, arguments, result)
    plan = agent.task.plan
    if result.success and plan is not None and plan.candidates:
        created_at = agent.pending_actions.now()
        ttl = agent.dialogue_config.candidate_ttl_seconds
        snapshot = CandidateSnapshot(
            candidate_set_id=uuid4().hex,
            constraint_revision=agent.task.revision,
            created_at=created_at,
            presented_turn_id=None,
            expires_at=created_at + ttl,
            presented_poi_ids=(),
        )
        plan.candidate_snapshot = snapshot
        return ""
    _clear(agent, result.error or "NO_RESULTS")
    agent.task.transition(TaskStatus.AWAITING_INPUT, result.error or "NO_RESULTS")
    return (
        "本次搜索没有找到可用的休息地点候选，请调整条件后重新查询。"
        if result.error == "NO_RESULTS"
        else f"本次搜索未能完成（{result.error or 'UNKNOWN'}），请稍后重试。"
    )


def present_candidates(agent: VehicleAgent) -> str:
    plan = agent.task.plan
    if plan is None or plan.candidate_snapshot is None or not plan.candidates:
        return "目前没有可展示的休息地点候选，请重新查询。"
    snapshot = plan.candidate_snapshot
    lines = ["找到以下模拟休息地点："]
    ids = []
    for index, candidate in enumerate(plan.candidates, start=1):
        poi_id = candidate.get("poi_id")
        if not isinstance(poi_id, str) or not poi_id:
            continue
        ids.append(poi_id)
        details = []
        distance = candidate.get("distance_km")
        eta = candidate.get("eta_minutes")
        details.append(
            f"{distance} 公里" if isinstance(distance, int | float) else "距离未提供"
        )
        details.append(
            f"预计 {eta} 分钟" if isinstance(eta, int | float) else "预计时间未提供"
        )
        lines.append(
            f"{index}. {agent.plan_flow.display_name(candidate)}（{'，'.join(details)}）"
        )
    if not ids:
        return "目前没有可展示的休息地点候选，请重新查询。"
    plan.candidate_snapshot = replace(
        snapshot,
        presented_turn_id=agent.dialogue_session.turn_id,
        presented_poi_ids=tuple(ids),
    )
    agent.task.last_presented_candidate_set_id = snapshot.candidate_set_id
    agent.task.transition(TaskStatus.AWAITING_INPUT, "CANDIDATES_PRESENTED")
    return "\n".join(lines) + "\n请按序号或候选名称选择；导航前仍需单独确认。"


def select_candidate(agent: VehicleAgent, candidate: dict) -> str:
    if not candidate_is_current(agent) or agent.task.unresolved_constraints:
        return "当前候选已失效或仍有未解决条件，请重新查询后选择。"
    candidate_id = candidate.get("poi_id")
    plan = agent.task.plan
    matches = [item for item in plan.candidates if item.get("poi_id") == candidate_id]
    if len(matches) != 1:
        return "该地点不属于当前有效候选，请重新选择。"
    pending = agent.pending_actions.get()
    if (
        pending is not None
        and pending.tool_name == "start_navigation"
        and pending.arguments.get("poi_id") == candidate_id
    ):
        return f"{agent.plan_flow.display_name(matches[0])} 的导航操作仍待确认，请使用当前确认入口。"
    agent.plan_flow._stage_candidate(
        agent, plan, matches[0], reason="用户从当前展示的候选中明确选择。"
    )
    return (
        f"已选择 {agent.plan_flow.display_name(matches[0])}；导航尚未执行，请明确确认。"
    )


def validate_pending_candidate(agent: VehicleAgent, action_id: str) -> bool:
    if not agent.dialogue_config.enabled:
        return True
    pending = agent.pending_actions.get()
    saved = agent.task.pending_action
    action = pending.to_dict() if pending is not None else saved
    if (
        not isinstance(action, dict)
        or action.get("action_id") != action_id
        or action.get("tool_name") != "start_navigation"
    ):
        return True
    metadata = action.get("metadata", {})
    plan = agent.task.plan
    snapshot = plan.candidate_snapshot if plan is not None else None
    valid = (
        candidate_is_current(agent)
        and snapshot is not None
        and action.get("arguments", {}).get("poi_id") == plan.selected_poi_id
        and metadata.get("candidate_set_id") == snapshot.candidate_set_id
        and metadata.get("constraint_revision") == agent.task.revision
        and metadata.get("expires_at") == snapshot.expires_at
        and agent.pending_actions.now() < snapshot.expires_at
    )
    if valid:
        return True
    _clear(agent, "INVALID_CANDIDATE_CONFIRMATION")
    agent.task.transition(TaskStatus.AWAITING_INPUT, "INVALID_CANDIDATE_CONFIRMATION")
    return False

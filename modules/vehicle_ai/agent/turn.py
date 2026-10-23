"""Bounded model/tool loop. No automatic replay of uncertain writes."""

import json
from typing import TYPE_CHECKING

from modules.vehicle_ai.agent.budget import BudgetExceeded
from modules.vehicle_ai.agent.session import record
from modules.vehicle_ai.agent.target_resolution import (
    TargetResolution,
    target_resolution_message,
)
from modules.vehicle_ai.agent.task_state import TaskStatus

if TYPE_CHECKING:
    from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent


def reconcile(agent: "VehicleAgent", name: str, result):
    """Read back state without treating matching current state as execution proof."""
    if name in {"set_ac", "set_temperature"}:
        query = "get_climate_status"
    elif name in {"play_music", "pause_music", "set_volume"}:
        query = "get_media_status"
    else:
        query = "get_vehicle_status"
    try:
        # Reserve one deterministic read-only call for uncertain-write recovery.
        observed = agent.tool_registry.execute(
            query, {}, user_intent=agent.current_user_intent
        )
        agent.task.reconciliation = {"tool": query, "result": observed.to_dict()}
        agent.task.tool_results.append(observed.to_dict())
        record(
            agent,
            "tool_result",
            source=query,
            quality="TOOL_RESULT",
            result=observed.to_dict(),
        )
    except BudgetExceeded as exc:
        agent.task.reconciliation = {"tool": query, "error": str(exc), "deferred": True}
    agent.task.last_tool_result = result.to_dict()
    agent.task.transition(TaskStatus.AWAITING_INPUT, "WRITE_OUTCOME_UNKNOWN")
    return "操作结果尚不确定，已停止自动重试；请核对车机状态后再决定下一步。"


def _candidate_display_name(candidate: dict | None) -> str | None:
    if candidate is None:
        return None
    for key in ("display_name_zh", "name", "poi_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _target_resolution_reply(
    agent: "VehicleAgent", resolution: TargetResolution
) -> str:
    candidate = resolution.candidate
    selected_id = (
        candidate.get("poi_id")
        if resolution.status == "matched" and candidate is not None
        else None
    )
    pending = agent.pending_actions.get()
    pending_created = bool(
        pending is not None
        and pending.tool_name == "start_navigation"
        and pending.arguments.get("poi_id") == selected_id
        and selected_id is not None
    )
    record(
        agent,
        "target_resolution",
        source="search_nearby_rest_area",
        quality="TOOL_RESULT",
        target=resolution.target,
        status=resolution.status,
        candidate_ids=list(resolution.candidate_ids),
        selected_id=selected_id,
        selected_display_name=_candidate_display_name(candidate),
        pending_action_id=(
            pending.action_id if pending is not None and pending_created else None
        ),
    )
    return target_resolution_message(resolution, pending_created=pending_created)


def _append_tool_result_message(
    messages: list[dict], call_id: str, result: dict
) -> None:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        }
    )


def _append_skipped_tool_call_messages(messages: list[dict], calls) -> None:
    for call in calls:
        _append_tool_result_message(
            messages,
            call.id,
            {
                "success": False,
                "error": "TARGET_RESOLUTION_COMPLETE",
                "message": "目标搜索结果已处理；后续工具调用未执行。",
            },
        )


def run_turn(agent, user_text: str, messages: list, target: str | None) -> str:
    """保留旧调用入口，实际编排由 LangGraph 完成。"""
    return agent.workflow.run_prepared(user_text, messages, target)

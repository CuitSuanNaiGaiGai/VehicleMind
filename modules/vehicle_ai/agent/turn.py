"""Bounded model/tool loop. No automatic replay of uncertain writes."""

import json
from typing import TYPE_CHECKING

from modules.vehicle_ai.agent.budget import BudgetExceeded, TurnBudget
from modules.vehicle_ai.agent.plan import PlanTransitionError
from modules.vehicle_ai.agent.session import record
from modules.vehicle_ai.agent.target_resolution import (
    TargetResolution,
    target_resolution_message,
    target_search_failure_message,
)
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.tools.validation import valid_arguments

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


def run_turn(
    agent: "VehicleAgent", user_text: str, messages: list, target: str | None
) -> str:
    budget = TurnBudget(
        agent.turn_timeout_seconds, agent.max_tool_calls, agent.budget_clock
    )
    tools = agent.tool_registry.llm_schemas()

    def stop(reason, text):
        agent.task.transition(TaskStatus.FAILED, reason)
        return agent._record_final_response(user_text, text)

    try:
        for _ in range(agent.max_tool_rounds):
            remaining = budget.remaining()
            try:
                response = agent.llm.chat_with_timeout(
                    messages, tools, timeout_seconds=remaining
                )
            except Exception as exc:
                reason = (
                    "MODEL_TIMEOUT"
                    if "timeout" in type(exc).__name__.lower()
                    else "MODEL_ERROR"
                )
                return stop(
                    reason, "模型请求失败，本轮已停止；已执行的操作不会自动撤销。"
                )
            budget.remaining()
            if not response.tool_calls:
                text = response.content or ""
                if not text.strip():
                    return stop("EMPTY_RESPONSE", "模型未返回有效回复，本轮已停止。")
                current = agent.pending_actions.get()
                if target is not None and (
                    current is None or current.tool_name != "start_navigation"
                ):
                    agent.task.transition(
                        TaskStatus.AWAITING_INPUT, "TARGET_SEARCH_REQUIRED"
                    )
                    return agent._record_final_response(
                        user_text,
                        "尚未执行本次目标搜索，未创建待确认导航；请重新发起搜索。",
                    )
                agent.plan_flow.finish_turn(agent, text)
                current = agent.pending_actions.get()
                agent.task.finish(current is not None)
                if (
                    agent.task.status is TaskStatus.COMPLETED
                    and text.rstrip().endswith(("?", "？"))
                    and not agent.task.last_tool_result
                ):
                    agent.task.transition(
                        TaskStatus.AWAITING_INPUT, "CLARIFICATION_REQUESTED"
                    )
                if agent.task.status is TaskStatus.FAILED:
                    text = "工具执行未成功，任务未完成；请检查失败原因后重试。"
                return agent._record_final_response(user_text, text)

            messages.append(agent._assistant_tool_message(response))
            for call_index, call in enumerate(response.tool_calls):
                try:
                    raw = json.loads(call.arguments_json)
                    if (
                        not isinstance(raw, dict)
                        or not isinstance(call.arguments, dict)
                        or raw != call.arguments
                    ):
                        raise ValueError("arguments must be an object")
                except (ValueError, TypeError):
                    return stop(
                        "INVALID_ARGUMENTS",
                        "工具参数格式不正确，本轮已停止，未执行该操作。",
                    )
                arguments = agent._ground_tool_arguments(call.name, call.arguments)
                try:
                    definition = agent.tool_registry.get(call.name)
                    schema = definition.parameters
                except KeyError:
                    definition = None
                    schema = None
                if schema is not None and not valid_arguments(arguments, schema):
                    return stop(
                        "INVALID_ARGUMENTS",
                        "工具参数不符合要求，本轮已停止，未执行该操作。",
                    )
                try:
                    plan_error = agent.plan_flow.before_tool(
                        agent, call.name, arguments, target=target
                    )
                except PlanTransitionError:
                    return agent._record_final_response(
                        user_text,
                        "当前计划已达到步骤预算，已停止且未执行后续操作。",
                    )
                if plan_error is not None:
                    return agent._record_final_response(user_text, plan_error)
                budget.claim(call.name, arguments)
                pending = agent.pending_actions.get()
                if (
                    definition is not None
                    and definition.requires_confirmation
                    and (
                        call.name != "start_navigation"
                        or (pending and pending.tool_name == "start_navigation")
                        or agent.plan_flow.candidate_for(
                            agent, str(arguments.get("poi_id", ""))
                        )
                    )
                ):
                    candidate = agent.plan_flow.candidate_for(
                        agent, str(arguments.get("poi_id", ""))
                    )
                    agent.confirmations.stage(
                        call.name,
                        arguments,
                        user_intent=user_text,
                        display_text=(
                            f"确认导航至{agent.plan_flow.display_name(candidate)}"
                            if call.name == "start_navigation" and candidate
                            else None
                        ),
                        metadata={"candidate": candidate} if candidate else None,
                    )
                result = agent.tool_registry.execute(
                    call.name, arguments, user_intent=user_text
                )
                if call.name == "play_music" and result.success and result.policy:
                    warning = next(
                        (
                            warning
                            for warning in result.policy["warnings"]
                            if "停车休息" in warning
                        ),
                        "",
                    )
                    agent._turn_music_paused = False
                    if warning:
                        agent._turn_music_warning = warning
                elif call.name == "pause_music" and result.success:
                    agent._turn_music_paused = True
                agent.task.last_tool_result = result.to_dict()
                agent.task.tool_results.append(result.to_dict())
                record(
                    agent,
                    "tool_result",
                    source=call.name,
                    quality="TOOL_RESULT",
                    arguments=arguments,
                    result=result.to_dict(),
                )
                retry_allowed = False
                if (
                    definition is not None
                    and definition.read_only
                    and result.data.get("retryable")
                ):
                    try:
                        budget.claim_retry(call.name, arguments)
                    except BudgetExceeded as exc:
                        agent.plan_flow.stop_current(agent, str(exc))
                        raise
                    retry_allowed = agent.plan_flow.claim_read_retry(
                        agent, call.name, result
                    )
                resolution: TargetResolution | None = None
                try:
                    resolution = agent.plan_flow.observe_tool(
                        agent,
                        call.name,
                        arguments,
                        result,
                        target=target,
                        allow_retry=retry_allowed,
                    )
                except PlanTransitionError:
                    _append_tool_result_message(messages, call.id, result.to_dict())
                    _append_skipped_tool_call_messages(
                        messages, response.tool_calls[call_index + 1 :]
                    )
                    return agent._record_final_response(
                        user_text,
                        "当前计划已达到步骤预算，已停止继续执行。",
                    )
                did_retry = False
                if retry_allowed:
                    did_retry = True
                    retry_result = agent.tool_registry.execute(
                        call.name, arguments, user_intent=user_text
                    )
                    agent.task.last_tool_result = retry_result.to_dict()
                    agent.task.tool_results.append(retry_result.to_dict())
                    record(
                        agent,
                        "tool_result",
                        source=call.name,
                        quality="TOOL_RESULT",
                        arguments=arguments,
                        attempt=2,
                        result=retry_result.to_dict(),
                    )
                    result = retry_result
                    try:
                        resolution = agent.plan_flow.observe_tool(
                            agent, call.name, arguments, result, target=target
                        )
                    except PlanTransitionError:
                        _append_tool_result_message(messages, call.id, result.to_dict())
                        _append_skipped_tool_call_messages(
                            messages, response.tool_calls[call_index + 1 :]
                        )
                        return agent._record_final_response(
                            user_text,
                            "当前计划已达到步骤预算，已停止继续执行。",
                        )
                if result.data.get("outcome_unknown"):
                    text = reconcile(agent, call.name, result)
                    return agent._record_final_response(user_text, text)
                agent._update_action_state(call.name, result)
                budget.remaining()
                _append_tool_result_message(messages, call.id, result.to_dict())
                target_search_finished = (
                    target is not None
                    and call.name == "search_nearby_rest_area"
                    and (resolution is not None or not retry_allowed or did_retry)
                )
                if target_search_finished:
                    _append_skipped_tool_call_messages(
                        messages, response.tool_calls[call_index + 1 :]
                    )
                    if resolution is not None:
                        answer = _target_resolution_reply(agent, resolution)
                    else:
                        if agent.task.status is not TaskStatus.AWAITING_INPUT:
                            agent.task.transition(
                                TaskStatus.AWAITING_INPUT,
                                result.error or "TARGET_SEARCH_NOT_COMPLETED",
                            )
                        answer = target_search_failure_message(
                            result.error or "TARGET_SEARCH_NOT_COMPLETED"
                        )
                    return agent._record_final_response(user_text, answer)
                pending = agent.pending_actions.get()
                if (
                    result.error == "CONFIRMATION_REQUIRED"
                    and pending
                    and pending.tool_name == call.name
                    and dict(pending.arguments) == arguments
                ):
                    agent.task.transition(TaskStatus.AWAITING_CONFIRMATION)
                    return agent._record_final_response(
                        user_text, "车机操作已准备好，尚未执行，待确认后才会执行。"
                    )
    except BudgetExceeded as exc:
        return stop(
            str(exc),
            "已达到本轮时间、工具次数或重复调用限制，停止继续执行；已执行的操作不会自动撤销。",
        )
    return stop("ROUND_LIMIT", "本次请求涉及过多连续工具调用，已停止执行。")

"""Bounded model/tool loop. No automatic replay of uncertain writes."""

import json
from typing import TYPE_CHECKING

from modules.vehicle_ai.agent.budget import BudgetExceeded, TurnBudget
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.agent.session import record
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
                    return stop(
                        "TARGET_NOT_FOUND",
                        "未找到与新目标匹配的地点，未创建待确认导航。",
                    )
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
            for call in response.tool_calls:
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
                    schema = agent.tool_registry.get(call.name).parameters
                except KeyError:
                    schema = None
                if schema is not None and not valid_arguments(arguments, schema):
                    return stop(
                        "INVALID_ARGUMENTS",
                        "工具参数不符合要求，本轮已停止，未执行该操作。",
                    )
                budget.claim(call.name, arguments)
                pending = agent.pending_actions.get()
                if call.name != "start_navigation" or (
                    pending and pending.tool_name == "start_navigation"
                ):
                    agent.confirmations.stage(
                        call.name, arguments, user_intent=user_text
                    )
                result = agent.tool_registry.execute(
                    call.name, arguments, user_intent=user_text
                )
                if call.name == "play_music" and result.success and result.policy:
                    agent._turn_music_warning = next(
                        (
                            warning
                            for warning in result.policy["warnings"]
                            if "停车休息" in warning
                        ),
                        "",
                    )
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
                if result.data.get("outcome_unknown"):
                    text = reconcile(agent, call.name, result)
                    return agent._record_final_response(user_text, text)
                agent._update_action_state(call.name, result, target)
                budget.remaining()
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            result.to_dict(), ensure_ascii=False, default=str
                        ),
                    }
                )
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

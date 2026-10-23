"""逐个工具的校验、执行、重试和观察节点，复用既有业务规则。"""

from __future__ import annotations

import json
from dataclasses import asdict
from copy import deepcopy
from langgraph.types import Overwrite

from modules.vehicle_ai.agent.graph import budget as budgets
from modules.vehicle_ai.agent.graph.nodes import ConversationNodes
from modules.vehicle_ai.agent.session import record
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.agent.target_resolution import (
    target_search_failure_message,
)
from modules.vehicle_ai.agent.turn import reconcile, _target_resolution_reply
from modules.vehicle_ai.agent import turn as turn_support
from modules.vehicle_ai.agent.plan import PlanTransitionError
from modules.vehicle_ai.tools.base import ToolResult
from modules.vehicle_ai.tools.validation import valid_arguments


def tool_message(call: dict, result: dict) -> dict:
    messages: list[dict] = []
    turn_support._append_tool_result_message(messages, call["id"], result)
    return messages[0]


class ToolNodes(ConversationNodes):
    def validate_tool(self, state):
        call = state["tool_calls"][state["call_index"]]
        try:
            raw = json.loads(call["arguments_json"])
            if (
                type(raw) is not dict
                or type(call["arguments"]) is not dict
                or raw != call["arguments"]
            ):
                raise ValueError("invalid arguments")
        except (TypeError, ValueError):
            return self.fail(
                "INVALID_ARGUMENTS", "工具参数格式不正确，本轮已停止，未执行该操作。"
            )
        arguments = self.agent._ground_tool_arguments(call["name"], call["arguments"])
        try:
            schema = self.agent.tool_registry.get(call["name"]).parameters
        except KeyError:
            schema = None
        if schema is not None and not valid_arguments(arguments, schema):
            return self.fail(
                "INVALID_ARGUMENTS", "工具参数不符合要求，本轮已停止，未执行该操作。"
            )
        return {
            "arguments": arguments,
            "route": "prepare_tool",
            "retry_allowed": False,
            "did_retry": False,
            "resolution": None,
        }

    def prepare_tool(self, state):
        call = state["tool_calls"][state["call_index"]]
        name, arguments = call["name"], state["arguments"]
        blocked = self.agent.plan_flow.before_tool(
            self.agent, name, arguments, target=state.get("target")
        )
        if blocked:
            return {"route": "respond", "response_text": blocked}
        budget = budgets.load(self.agent, state)
        budget.claim(name, arguments)
        try:
            definition = self.agent.tool_registry.get(name)
        except KeyError:
            definition = None
        pending = self.agent.pending_actions.get()
        candidate = self.agent.plan_flow.candidate_for(
            self.agent, str(arguments.get("poi_id", ""))
        )
        if (
            definition is not None
            and definition.requires_confirmation
            and (
                name != "start_navigation"
                or (pending and pending.tool_name == name)
                or candidate
            )
        ):
            self.agent.confirmations.stage(
                name,
                arguments,
                user_intent=state["input_text"],
                display_text=f"确认导航至{self.agent.plan_flow.display_name(candidate)}"
                if name == "start_navigation" and candidate
                else None,
                metadata={"candidate": candidate} if candidate else None,
            )
        return {"budget": budgets.dump(budget), "route": "execute_tool"}

    def execute_tool(self, state):
        budgets.load(self.agent, state).remaining()
        call = state["tool_calls"][state["call_index"]]
        result = self.workflow.execute_tool(state, call, attempt=1)
        return {"tool_result": result.to_dict(), "route": "observe_tool"}

    def _record_result(self, name, arguments, result, attempt=1):
        if name == "play_music" and result.success and result.policy:
            warning = next(
                (
                    warning
                    for warning in result.policy["warnings"]
                    if "停车休息" in warning
                ),
                "",
            )
            self.agent._turn_music_paused = False
            if warning:
                self.agent._turn_music_warning = warning
        elif name == "pause_music" and result.success:
            self.agent._turn_music_paused = True
        self.agent.task.last_tool_result = result.to_dict()
        self.agent.task.tool_results.append(result.to_dict())
        record(
            self.agent,
            "tool_result",
            source=name,
            quality="TOOL_RESULT",
            arguments=arguments,
            result=result.to_dict(),
            **({"attempt": attempt} if attempt > 1 else {}),
        )

    def observe_tool(self, state):
        call = state["tool_calls"][state["call_index"]]
        name, arguments = call["name"], state["arguments"]
        result = ToolResult(**state["tool_result"])
        self._record_result(
            name, arguments, result, attempt=2 if state.get("did_retry") else 1
        )
        try:
            definition = self.agent.tool_registry.get(name)
        except KeyError:
            definition = None
        retry_allowed = False
        budget = budgets.load(self.agent, state)
        if (
            not state.get("did_retry")
            and definition
            and definition.read_only
            and result.data.get("retryable")
        ):
            budget.claim_retry(name, arguments)
            retry_allowed = self.agent.plan_flow.claim_read_retry(
                self.agent, name, result
            )
        try:
            resolution = self.agent.plan_flow.observe_tool(
                self.agent,
                name,
                arguments,
                result,
                target=state.get("target"),
                allow_retry=retry_allowed,
            )
        except PlanTransitionError:
            messages = deepcopy(state["messages"])
            turn_support._append_tool_result_message(
                messages, call["id"], result.to_dict()
            )
            for skipped in state["tool_calls"][state["call_index"] + 1 :]:
                turn_support._append_tool_result_message(
                    messages,
                    skipped["id"],
                    {
                        "success": False,
                        "error": "TARGET_RESOLUTION_COMPLETE",
                        "message": "目标搜索结果已处理；后续工具调用未执行。",
                    },
                )
            return {
                "messages": Overwrite(messages),
                "response_text": "当前计划已达到步骤预算，已停止继续执行。",
                "route": "respond",
            }
        if retry_allowed:
            return {
                "budget": budgets.dump(budget),
                "route": "retry_tool",
                "retry_allowed": True,
            }
        if result.data.get("outcome_unknown"):
            return {"route": "reconcile_write"}
        self.agent._update_action_state(name, result)
        budget.remaining()
        message = tool_message(call, result.to_dict())
        index = state["call_index"] + 1
        if state.get("target") is not None and name == "search_nearby_rest_area":
            if resolution is not None:
                answer = _target_resolution_reply(self.agent, resolution)
            else:
                if self.agent.task.status is not TaskStatus.AWAITING_INPUT:
                    self.agent.task.transition(
                        TaskStatus.AWAITING_INPUT,
                        result.error or "TARGET_SEARCH_NOT_COMPLETED",
                    )
                answer = target_search_failure_message(
                    result.error or "TARGET_SEARCH_NOT_COMPLETED"
                )
            return {
                "messages": [message],
                "response_text": answer,
                "route": "respond",
                "resolution": asdict(resolution) if resolution else None,
            }
        pending = self.agent.pending_actions.get()
        if (
            result.error == "CONFIRMATION_REQUIRED"
            and pending
            and pending.tool_name == name
            and dict(pending.arguments) == arguments
        ):
            self.agent.task.transition(TaskStatus.AWAITING_CONFIRMATION)
            return {
                "messages": [message],
                "response_text": "车机操作已准备好，尚未执行，待确认后才会执行。",
                "route": "respond",
            }
        return {
            "messages": [message],
            "call_index": index,
            "budget": budgets.dump(budget),
            "route": "validate_tool" if index < len(state["tool_calls"]) else "decide",
        }

    def retry_tool(self, state):
        budgets.load(self.agent, state).remaining()
        call = state["tool_calls"][state["call_index"]]
        result = self.workflow.execute_tool(state, call, attempt=2)
        return {
            "tool_result": result.to_dict(),
            "did_retry": True,
            "route": "observe_tool",
        }

    def reconcile_write(self, state):
        call = state["tool_calls"][state["call_index"]]
        text = reconcile(self.agent, call["name"], ToolResult(**state["tool_result"]))
        return {"response_text": text, "route": "respond"}

    def confirm_action(self, state):
        event = state["event"]
        action_id = event.get("action_id")
        if type(action_id) is not str or not action_id:
            result = ToolResult(
                False, "确认输入缺少有效 action_id。", error="INVALID_CONFIRMATION"
            )
        elif event["kind"] == "reject":
            result = self.agent._reject_pending_business(action_id)
        else:
            result = self.workflow.execute_confirmation(
                action_id, state.get("pending_action")
            )
        return {
            "confirmation_result": result.to_dict(),
            "final_response": result.message,
            "route": "await_confirmation"
            if self.agent.pending_actions.get()
            else "end",
        }

    def invalid_resume(self, state):
        result = ToolResult(
            False, "恢复输入无效，操作尚未执行。", error="INVALID_CONFIRMATION"
        )
        return {
            "confirmation_result": result.to_dict(),
            "final_response": result.message,
            "route": "await_confirmation"
            if self.agent.pending_actions.get()
            else "end",
        }

"""用户接入、上下文、模型决策和响应节点，路由显式交给 LangGraph。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import time

from langgraph.types import Overwrite, interrupt

from modules.vehicle_ai.agent.graph import budget as budgets
from modules.vehicle_ai.agent.graph.business import capture, restore, plain
from modules.vehicle_ai.agent.prompts import SYSTEM_PROMPT
from modules.vehicle_ai.agent.session import evidence_message, trip_memory_message
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.agent.budget import BudgetExceeded
from modules.vehicle_ai.agent.plan import PlanTransitionError


class ConversationNodes:
    def __init__(self, agent, workflow):
        self.agent = agent
        self.workflow = workflow

    def bind(self, name, handler):
        def node(state):
            self.workflow.storage.renew_session(
                self.workflow.thread_id, self.agent.turn_timeout_seconds + 30
            )
            restore(self.agent, state)
            started = time.perf_counter()
            try:
                update = handler(state)
            except BudgetExceeded as error:
                update = self.fail(
                    str(error),
                    "已达到本轮时间、工具次数或重复调用限制，停止继续执行；已执行的操作不会自动撤销。",
                )
            except PlanTransitionError:
                update = {
                    "response_text": "当前计划已达到步骤预算，已停止继续执行。",
                    "route": "respond",
                }
            trace = [
                *state.get("graph_trace", []),
                {
                    "node": name,
                    "turn_id": state.get("turn_id"),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "route": update.get("route"),
                    "task_status": self.agent.task.status.value,
                },
            ][-self.agent.max_task_trace_events :]
            return plain(
                {
                    **capture(self.agent),
                    **update,
                    "graph_trace": trace,
                    "simulated_vehicle": dict(
                        self.agent.context_manager.get_context().vehicle.__dict__
                    ),
                }
            )

        return node

    def fail(self, reason: str, text: str) -> dict:
        self.agent.task.transition(TaskStatus.FAILED, reason)
        return {"route": "respond", "stop_reason": reason, "response_text": text}

    def receive_request(self, state):
        event = state["event"]
        if event["kind"] in {"confirm", "reject"}:
            return {"route": "confirm_action"}
        user_text = event["text"].strip()
        if not user_text:
            return {"route": "respond", "response_text": ""}
        budget, deadline = budgets.create(self.agent)
        if event["kind"] == "prepared":
            return {
                "input_text": user_text,
                "messages": Overwrite(event["messages"]),
                "target": event.get("target"),
                "budget": budget,
                "deadline_at": deadline,
                "model_rounds": 0,
                "route": "decide",
            }
        if self.agent.dialogue_config.enabled and self.agent.dialogue.handles(
            user_text
        ):
            self.agent.dialogue.begin_turn(user_text)
            return {
                "input_text": user_text,
                "debug": event.get("debug", False),
                "budget": budget,
                "deadline_at": deadline,
                "route": "dialogue_interpret",
            }
        result = self.agent._prepare_request(user_text, debug=event.get("debug", False))
        return {
            "input_text": user_text,
            "debug": event.get("debug", False),
            "budget": budget,
            "deadline_at": deadline,
            "target": result.get("target") if isinstance(result, dict) else None,
            "response_text": result if isinstance(result, str) else "",
            "route": "respond" if isinstance(result, str) else "build_context",
            "model_rounds": 0,
            "tool_calls": [],
            "call_index": 0,
            "tool_result": None,
            "confirmation_result": None,
            "stop_reason": None,
        }

    def dialogue_interpret(self, state):
        budget = budgets.load(self.agent, state)
        proposal = self.agent.dialogue.interpret(state["input_text"], budget)
        budget.remaining()
        return {
            "dialogue_proposal": proposal,
            "budget": budgets.dump(budget),
            "route": "dialogue_reduce",
        }

    def dialogue_reduce(self, state):
        proposal = state["dialogue_proposal"]
        decision = self.agent.dialogue.reduce(proposal["proposal"], proposal["valid"])
        return {"dialogue_decision": decision, "route": "dialogue_execute"}

    def dialogue_execute(self, state):
        budget = budgets.load(self.agent, state)
        answer = self.agent.dialogue.execute_decision(
            state["dialogue_decision"], budget
        )
        budget.remaining()
        return {
            "response_text": answer,
            "budget": budgets.dump(budget),
            "route": "respond",
        }

    def build_context(self, state):
        agent = self.agent
        message = agent._context_message(
            f"{agent.task.goal} {state['input_text']}", state["debug"]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            message,
            agent._pending_action_message(),
            evidence_message(agent),
            *agent.history,
            {"role": "user", "content": state["input_text"]},
        ]
        memory = trip_memory_message(agent.tool_registry.names())
        if memory:
            messages.insert(3, memory)
        return {
            "messages": Overwrite(messages),
            "context": {"prompt": message["content"]},
            "route": "decide",
        }

    def decide(self, state):
        if state["model_rounds"] >= self.agent.max_tool_rounds:
            return self.fail(
                "ROUND_LIMIT", "本次请求涉及过多连续工具调用，已停止执行。"
            )
        budget = budgets.load(self.agent, state)
        remaining = budget.remaining()
        try:
            response = self.agent.llm.chat_with_timeout(
                deepcopy(state["messages"]),
                self.agent.tool_registry.llm_schemas(),
                timeout_seconds=remaining,
            )
        except Exception as error:
            reason = (
                "MODEL_TIMEOUT"
                if "timeout" in type(error).__name__.lower()
                else "MODEL_ERROR"
            )
            return self.fail(
                reason, "模型请求失败，本轮已停止；已执行的操作不会自动撤销。"
            )
        budget.remaining()
        rounds = state["model_rounds"] + 1
        if not response.tool_calls:
            text = response.content or ""
            if not text.strip():
                return self.fail("EMPTY_RESPONSE", "模型未返回有效回复，本轮已停止。")
            current = self.agent.pending_actions.get()
            if state.get("target") is not None and (
                current is None or current.tool_name != "start_navigation"
            ):
                self.agent.task.transition(
                    TaskStatus.AWAITING_INPUT, "TARGET_SEARCH_REQUIRED"
                )
                return {
                    "route": "respond",
                    "response_text": "尚未执行本次目标搜索，未创建待确认导航；请重新发起搜索。",
                    "model_rounds": rounds,
                }
            self.agent.plan_flow.finish_turn(self.agent, text)
            current = self.agent.pending_actions.get()
            self.agent.task.finish(current is not None)
            if (
                self.agent.task.status is TaskStatus.COMPLETED
                and text.rstrip().endswith(("?", "？"))
                and not self.agent.task.last_tool_result
            ):
                self.agent.task.transition(
                    TaskStatus.AWAITING_INPUT, "CLARIFICATION_REQUESTED"
                )
            if self.agent.task.status is TaskStatus.FAILED:
                text = "工具执行未成功，任务未完成；请检查失败原因后重试。"
            return {"route": "respond", "response_text": text, "model_rounds": rounds}
        return {
            "messages": [self.agent._assistant_tool_message(response)],
            "tool_calls": [asdict(call) for call in response.tool_calls],
            "call_index": 0,
            "model_rounds": rounds,
            "route": "validate_tool",
        }

    def respond(self, state):
        text = state.get("response_text", "")
        if state.get("input_text"):
            text = self.agent._record_final_response(state["input_text"], text)
        pending = self.agent.pending_actions.get()
        # 停止批量调用时，为尚未执行的调用补齐协议消息，避免 checkpoint 悬空。
        messages = state.get("messages", [])
        completed = {
            message.get("tool_call_id")
            for message in messages
            if message.get("role") == "tool"
        }
        skipped = []
        for call in state.get("tool_calls", []):
            if call["id"] not in completed:
                skipped.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(
                            {
                                "success": False,
                                "error": state.get("stop_reason") or "TURN_STOPPED",
                                "message": "本轮已停止，该工具调用未执行。",
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
        return {
            "messages": skipped,
            "final_response": text,
            "route": "await_confirmation" if pending else "end",
        }

    def await_confirmation(self, state):
        # 此节点中断前不执行写操作；恢复时只重新展示 checkpoint 绑定的动作。
        event = interrupt(
            {
                "kind": "confirmation_required",
                "thread_id": self.workflow.thread_id,
                "action": state.get("pending_action"),
                "message": state.get("final_response", ""),
            }
        )
        if type(event) is not dict or event.get("kind") not in {
            "chat",
            "confirm",
            "reject",
            "prepared",
        }:
            return {"event": {"kind": "invalid"}, "route": "invalid_resume"}
        return {
            "event": event,
            "route": "receive_request"
            if event["kind"] in {"chat", "prepared"}
            else "confirm_action",
        }

"""真实 LangGraph 编排入口：checkpoint、会话隔离、条件边和人工确认。"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from modules.vehicle_ai.agent.graph.business import capture, restore
from modules.vehicle_ai.agent.graph.state import AgentGraphState
from modules.vehicle_ai.agent.graph.storage import GraphStorage
from modules.vehicle_ai.agent.graph.tool_nodes import ToolNodes
from modules.vehicle_ai.agent.task_state import TaskStatus
from modules.vehicle_ai.agent.turn import reconcile
from modules.vehicle_ai.tools.base import ToolResult


class VehicleAgentWorkflow:
    def __init__(
        self,
        agent,
        checkpoint_path: str | Path,
        thread_id: str | None,
        recursion_limit: int,
    ):
        if type(recursion_limit) is not int or recursion_limit < 8:
            raise ValueError("graph_recursion_limit must be an integer >= 8")
        self.agent = agent
        self.thread_id = thread_id or uuid4().hex
        if not self.thread_id.strip():
            raise ValueError("thread_id must not be empty")
        self.recursion_limit = recursion_limit
        self.storage = GraphStorage(checkpoint_path)
        self.lock = RLock()
        # 监听器只属于本次运行，不进入 checkpoint，也不参与执行授权。
        self.stream_listener: Callable[[str, dict], None] | None = None
        self.nodes = ToolNodes(agent, self)
        self.graph = self._build()
        snapshot = self.graph.get_state(self.config)
        if snapshot.values:
            restore(agent, snapshot.values)
            # 仅恢复模拟车机后端；驾驶员和道路旧观测不作为当前传感器事实。
            vehicle = self.storage.latest_vehicle(
                self.thread_id
            ) or snapshot.values.get("simulated_vehicle")
            if vehicle:
                self._restore_vehicle(vehicle)
        self._checkpoint_id = self._version(snapshot)

    @staticmethod
    def _version(snapshot):
        return (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")

    def _refresh(self):
        snapshot = self.graph.get_state(self.config)
        version = self._version(snapshot)
        if version != self._checkpoint_id:
            restore(self.agent, snapshot.values)
            vehicle = self.storage.latest_vehicle(
                self.thread_id
            ) or snapshot.values.get("simulated_vehicle")
            if vehicle:
                self._restore_vehicle(vehicle)
            self._checkpoint_id = version
        return snapshot

    @property
    def config(self):
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.recursion_limit,
        }

    def _build(self):
        graph = StateGraph(AgentGraphState)
        names = (
            "receive_request",
            "build_context",
            "dialogue_interpret",
            "dialogue_reduce",
            "dialogue_execute",
            "decide",
            "validate_tool",
            "prepare_tool",
            "execute_tool",
            "observe_tool",
            "retry_tool",
            "reconcile_write",
            "respond",
            "await_confirmation",
            "confirm_action",
            "invalid_resume",
        )
        for name in names:
            graph.add_node(name, self.nodes.bind(name, getattr(self.nodes, name)))
        graph.add_edge(START, "receive_request")
        routes = {
            "receive_request": (
                "build_context",
                "decide",
                "dialogue_interpret",
                "respond",
                "confirm_action",
                "await_confirmation",
                "end",
            ),
            "build_context": ("decide", "respond"),
            "dialogue_interpret": ("dialogue_reduce", "respond"),
            "dialogue_reduce": ("dialogue_execute", "respond"),
            "dialogue_execute": ("respond",),
            "decide": ("validate_tool", "respond"),
            "validate_tool": ("prepare_tool", "respond"),
            "prepare_tool": ("execute_tool", "respond"),
            "execute_tool": ("observe_tool", "respond"),
            "observe_tool": (
                "retry_tool",
                "reconcile_write",
                "validate_tool",
                "decide",
                "respond",
            ),
            "retry_tool": ("observe_tool", "respond"),
            "reconcile_write": ("respond",),
            "respond": ("await_confirmation", "end"),
            "await_confirmation": (
                "receive_request",
                "confirm_action",
                "invalid_resume",
            ),
            "confirm_action": ("await_confirmation", "end", "respond"),
            "invalid_resume": ("await_confirmation", "end"),
        }
        for name, destinations in routes.items():
            graph.add_conditional_edges(
                name,
                lambda state: state["route"],
                {
                    destination: END if destination == "end" else destination
                    for destination in destinations
                },
            )
        return graph.compile(checkpointer=self.storage.checkpointer, name="VehicleMind")

    def _invoke(self, event: dict):
        snapshot = self._refresh()
        seed = {
            **capture(self.agent),
            "event": event,
            "turn_id": uuid4().hex,
            "input_text": "",
            "response_text": "",
            "final_response": "",
            "stop_reason": None,
            "confirmation_result": None,
            "tool_calls": [],
            "graph_trace": snapshot.values.get("graph_trace", []),
        }
        value: Command | dict[str, Any]
        if snapshot.next:
            if snapshot.next != ("await_confirmation",):
                raise RuntimeError(
                    "unfinished graph requires recover(), not a new request"
                )
            # 兼容工具调用者在会话视图中更新 PendingAction，再将其写回权威状态。
            self.graph.update_state(self.config, seed, as_node="respond")
            value = Command(resume=event)
        else:
            value = seed
        result = self._run(value)
        restore(self.agent, result)
        self._checkpoint_id = self._version(self.graph.get_state(self.config))
        return result

    def _run(self, value):
        try:
            if self.stream_listener is None:
                result = self.graph.invoke(value, self.config)
            else:
                result = self._stream(value)
        except GraphRecursionError:
            self.agent.task.transition(TaskStatus.FAILED, "GRAPH_RECURSION_LIMIT")
            self.agent.pending_actions.clear()
            text = "图执行达到步骤上限，已停止后续操作。"
            result = {
                **capture(self.agent),
                "final_response": text,
                "route": "end",
                "stop_reason": "GRAPH_RECURSION_LIMIT",
            }
            self.graph.update_state(self.config, result, as_node="confirm_action")
        return result

    def _stream(self, value):
        """消费原生节点事件及状态流，保持原公共接口的返回语义。"""
        listener = self.stream_listener
        result = {}
        for part in self.graph.stream(
            value, self.config, stream_mode=["tasks", "values"], version="v2"
        ):
            if part["type"] == "values":
                result = part["data"]
                if listener:
                    listener("state", result)
            elif part["type"] == "tasks" and listener:
                listener("node", part["data"])
        return result

    def chat(self, text: str, debug: bool = False) -> str:
        if not text.strip():
            return ""
        with (
            self.lock,
            self.storage.session(self.thread_id, self.agent.turn_timeout_seconds + 30),
        ):
            result = self._invoke({"kind": "chat", "text": text, "debug": debug})
            return result.get("final_response", "")

    def confirm(self, action_id: str) -> ToolResult:
        with (
            self.lock,
            self.storage.session(self.thread_id, self.agent.turn_timeout_seconds + 30),
        ):
            result = self._invoke({"kind": "confirm", "action_id": action_id})
            return ToolResult(**result["confirmation_result"])

    def reject(self, action_id: str) -> ToolResult:
        with (
            self.lock,
            self.storage.session(self.thread_id, self.agent.turn_timeout_seconds + 30),
        ):
            result = self._invoke({"kind": "reject", "action_id": action_id})
            return ToolResult(**result["confirmation_result"])

    def execute_tool(self, state, call, attempt: int) -> ToolResult:
        arguments = state["arguments"]
        try:
            definition = self.agent.tool_registry.get(call["name"])
        except KeyError:
            definition = None
        if (
            definition is None
            or definition.read_only
            or definition.requires_confirmation
        ):
            return self.agent.tool_registry.execute(
                call["name"], arguments, user_intent=self.agent.current_user_intent
            )
        operation_id = f"{state['turn_id']}:{state['model_rounds']}:{state['call_index']}:{attempt}"
        signature = json.dumps(
            [call["name"], arguments], sort_keys=True, ensure_ascii=False
        )

        def execute():
            result = self.agent.tool_registry.execute(
                call["name"], arguments, user_intent=self.agent.current_user_intent
            )
            return {
                "result": result.to_dict(),
                "vehicle": self.agent.context_manager.get_context().vehicle.__dict__,
            }

        payload = self.storage.run_once(
            self.thread_id, operation_id, signature, execute
        )
        return ToolResult(**payload["result"])

    def execute_confirmation(
        self, action_id: str, bound_action: dict | None
    ) -> ToolResult:
        payload = None
        operation_id = f"confirmation:{action_id}"
        if bound_action and bound_action.get("action_id") == action_id:
            signature = json.dumps(
                [bound_action["tool_name"], bound_action["arguments"]],
                sort_keys=True,
                ensure_ascii=False,
            )
            payload = self.storage.lookup(self.thread_id, operation_id, signature)
        if payload is not None:
            return self._confirmation_payload(payload, bound_action)
        pending = self.agent.pending_actions.get()
        if pending is None or pending.action_id != action_id:
            return self.agent._confirm_pending_business(action_id)
        signature = json.dumps(
            [pending.tool_name, dict(pending.arguments)],
            sort_keys=True,
            ensure_ascii=False,
        )

        def execute():
            result = self.agent._confirm_pending_business(action_id)
            return {
                "result": result.to_dict(),
                "business": capture(self.agent),
                "vehicle": self.agent.context_manager.get_context().vehicle.__dict__,
            }

        payload = self.storage.run_once(
            self.thread_id, f"confirmation:{action_id}", signature, execute
        )
        return self._confirmation_payload(payload, bound_action)

    def _confirmation_payload(self, payload, bound_action):
        if "business" in payload:
            restore(self.agent, payload["business"])
        elif payload["result"].get("data", {}).get("outcome_unknown"):
            self.agent.pending_actions.clear()
            self.agent.task.transition(
                TaskStatus.AWAITING_INPUT, "WRITE_OUTCOME_UNKNOWN"
            )
            self.agent.task.last_tool_result = payload["result"]
            self.agent.task.tool_results.append(payload["result"])
            if bound_action:
                reconcile(
                    self.agent,
                    bound_action["tool_name"],
                    ToolResult(**payload["result"]),
                )
        return ToolResult(**payload["result"])

    def recover(self) -> dict:
        """恢复中途停止的图；人工确认中断必须走 confirm/reject/chat 接口。"""
        with (
            self.lock,
            self.storage.session(self.thread_id, self.agent.turn_timeout_seconds + 30),
        ):
            snapshot = self._refresh()
            if snapshot.next == ("await_confirmation",):
                raise ValueError("pending confirmation requires an explicit decision")
            if snapshot.values.get("budget"):
                remaining = max(0.0, snapshot.values["deadline_at"] - time.time())
                budget = {
                    **snapshot.values["budget"],
                    "deadline": self.agent.budget_clock() + remaining,
                }
                self.graph.update_state(self.config, {"budget": budget})
            result = self._run(None)
            restore(self.agent, result)
            self._checkpoint_id = self._version(self.graph.get_state(self.config))
            return result

    def run_prepared(self, text, messages, target):
        # 旧入口的每次调用也必须拥有独立预算和执行标识。
        with (
            self.lock,
            self.storage.session(self.thread_id, self.agent.turn_timeout_seconds + 30),
        ):
            return self._invoke(
                {
                    "kind": "prepared",
                    "text": text,
                    "messages": messages,
                    "target": target,
                    "debug": False,
                }
            ).get("final_response", "")

    def _restore_vehicle(self, raw):
        from modules.vehicle_ai.context.enums import NavigationState

        values = deepcopy(raw)
        values.pop("updated_at", None)
        values.pop("source", None)
        values.pop("speed_kmh", None)
        values.pop("gear", None)
        values["navigation_state"] = NavigationState(values["navigation_state"])
        self.agent.context_manager.update_vehicle(**values)

    def reset_thread(self):
        with self.lock:
            self.thread_id = uuid4().hex
            self._checkpoint_id = None

    def close(self):
        self.storage.close()

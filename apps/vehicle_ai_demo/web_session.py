"""浏览器会话与真实 Graph 事件投影；确认仍交由既有安全执行层。"""

from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from threading import Condition, RLock, Thread
from uuid import uuid4

from apps.vehicle_ai_demo.web_model import WebDemoModel
from modules.vehicle_ai.agent.graph.business import plain
from modules.vehicle_ai.llm import build_llm_client
from modules.vehicle_ai.runtime import VehicleMindRuntime


PROVIDERS = {"scripted", "qwen", "glm"}
STATE_FIELDS = (
    "task",
    "pending_action",
    "history",
    "messages",
    "context",
    "arguments",
    "tool_calls",
    "tool_result",
    "budget",
    "model_rounds",
    "stop_reason",
    "graph_trace",
    "simulated_vehicle",
    "final_response",
    "confirmation_result",
)


def valid_session_id(value: str) -> bool:
    parts = value.split(".")
    return (
        len(parts) == 2
        and parts[0] in PROVIDERS
        and len(parts[1]) == 32
        and all(char in "0123456789abcdef" for char in parts[1])
    )


class WebSession:
    def __init__(self, session_id: str, database: Path, delay_seconds: float):
        self.session_id = session_id
        self.provider = session_id.split(".")[0]
        client = (
            WebDemoModel(delay_seconds)
            if self.provider == "scripted"
            else build_llm_client(self.provider)
        )
        self.runtime = VehicleMindRuntime(
            client,
            checkpoint_path=database,
            thread_id=f"web:{session_id}",
            enable_event_recommendations=False,
        )
        self.delay_seconds = delay_seconds
        self.condition = Condition(RLock())
        self.events: deque[dict] = deque(maxlen=256)
        self.sequence = 0
        self.busy = False
        self.active_node: str | None = None
        self.last_error: str | None = None
        self.worker: Thread | None = None
        self.token = uuid4().hex
        self.stream_id = uuid4().hex
        graph = self.runtime.agent.graph
        self.state = deepcopy(
            graph.get_state(self.runtime.agent.workflow.config).values
        )
        self.next_nodes = list(graph.get_state(self.runtime.agent.workflow.config).next)
        self.active_node = self.next_nodes[0] if self.next_nodes else None
        topology = graph.get_graph()
        self.topology = {
            "nodes": list(topology.nodes),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "conditional": edge.conditional,
                }
                for edge in topology.edges
            ],
        }
        self.runtime.agent.workflow.stream_listener = self.on_graph_event

    def view(self) -> dict:
        with self.condition:
            pending = deepcopy(self.state.get("pending_action"))
            if pending:
                pending["expires_at"] = (
                    pending["created_at"] + pending["expires_after_seconds"]
                )
                pending["expired"] = time.time() >= pending["expires_at"]
            return plain(
                {
                    "session_id": self.session_id,
                    "last_event_id": self.sequence,
                    "stream_id": self.stream_id,
                    "thread_id": self.runtime.agent.thread_id,
                    "provider": self.provider,
                    "demo_delay_seconds": self.delay_seconds
                    if self.provider == "scripted"
                    else None,
                    "busy": self.busy,
                    "active_node": self.active_node,
                    "last_error": self.last_error,
                    "next": self.next_nodes,
                    "pending_action": pending,
                    "conversation": [
                        {
                            "role": "user"
                            if event["kind"] == "user_request"
                            else "assistant",
                            "content": event.get("text")
                            if event["kind"] not in {"confirmation", "rejection"}
                            else event.get("result", {}).get("message"),
                        }
                        for event in self.state.get("agent_trace", [])
                        if event["kind"]
                        in {"user_request", "agent_reply", "confirmation", "rejection"}
                    ],
                    "state": {
                        key: deepcopy(self.state[key])
                        for key in STATE_FIELDS
                        if key in self.state
                    },
                    "graph": self.topology,
                }
            )

    def emit(self, kind: str, payload: dict) -> None:
        with self.condition:
            self.sequence += 1
            self.events.append(
                {
                    "id": self.sequence,
                    "kind": kind,
                    "timestamp": time.time(),
                    "payload": plain(deepcopy(payload)),
                }
            )
            self.condition.notify_all()

    def on_graph_event(self, kind: str, payload: dict) -> None:
        with self.condition:
            if kind == "state":
                self.state = deepcopy(payload)
                self.emit("state", self.view())
                return
            starting = "input" in payload
            waiting = bool(payload.get("interrupts"))
            self.active_node = payload["name"] if starting or waiting else None
            self.emit(
                "node",
                {
                    "name": payload["name"],
                    "execution_id": payload["id"],
                    "phase": "start"
                    if starting
                    else "waiting"
                    if waiting
                    else "error"
                    if payload.get("error")
                    else "finish",
                    "error": str(payload["error"]) if payload.get("error") else None,
                },
            )

    def submit(self, kind: str, payload: dict) -> None:
        if kind not in {"chat", "confirm", "reject", "recover"}:
            raise ValueError("不支持的操作类型")
        key = "text" if kind == "chat" else "action_id"
        if kind != "recover" and (
            not isinstance(payload.get(key), str)
            or not payload[key].strip()
            or len(payload[key]) > (4000 if kind == "chat" else 128)
        ):
            raise ValueError(f"缺少有效 {key}")
        with self.condition:
            if self.busy:
                raise RuntimeError("本会话正在执行，请等待当前操作结束")
            self.busy = True
            self.last_error = None
            self.active_node = None
            self.emit(
                "operation_start", {"operation": kind, "text": payload.get("text")}
            )
            self.worker = Thread(
                target=self._execute, args=(kind, payload), daemon=True
            )
            self.worker.start()

    def _execute(self, kind: str, payload: dict) -> None:
        try:
            agent = self.runtime.agent
            result: str | dict
            if kind == "chat":
                result = self.runtime.chat(payload["text"], debug=False)
            elif kind == "confirm":
                result = agent.confirm_pending(payload["action_id"]).to_dict()
            elif kind == "reject":
                result = agent.reject_pending(payload["action_id"]).to_dict()
            else:
                result = agent.workflow.recover().get("final_response", "")
            self.emit("response", {"operation": kind, "result": result})
        except Exception as error:
            # 不切换模型或重放操作，保留 checkpoint 供显式恢复。
            self.last_error = str(error)
            self.emit("operation_error", {"message": str(error)})
        finally:
            snapshot = self.runtime.agent.graph.get_state(
                self.runtime.agent.workflow.config
            )
            with self.condition:
                self.state = deepcopy(snapshot.values)
                self.next_nodes = list(snapshot.next)
                self.active_node = self.next_nodes[0] if self.next_nodes else None
                self.busy = False
                self.emit("operation_end", self.view())

    def close(self):
        if self.worker:
            self.worker.join()
        self.runtime.close()


class SessionRegistry:
    def __init__(self, database: Path, delay_seconds: float, provider: str):
        self.database, self.delay_seconds, self.provider = (
            database,
            delay_seconds,
            provider,
        )
        self.sessions: dict[str, WebSession] = {}
        self.lock = RLock()

    def new_id(self, provider: str | None = None) -> str:
        selected = provider or self.provider
        if selected not in PROVIDERS:
            raise ValueError("模型只能选择 scripted、qwen 或 glm")
        return f"{selected}.{uuid4().hex}"

    def get(self, session_id: str) -> WebSession:
        if not valid_session_id(session_id):
            raise ValueError("无效的会话 ID")
        with self.lock:
            if session_id not in self.sessions:
                if len(self.sessions) >= 64:
                    raise RuntimeError("演示服务会话已达上限，请重启服务")
                self.sessions[session_id] = WebSession(
                    session_id, self.database, self.delay_seconds
                )
            return self.sessions[session_id]

    def close(self):
        for session in self.sessions.values():
            session.close()

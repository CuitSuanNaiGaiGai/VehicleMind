"""LangGraph orchestration around the existing bounded VehicleAgent executor."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, Callable, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from modules.vehicle_ai.agent import VehicleAgent
from modules.vehicle_ai.events import EventPriority, EventType, VehicleEvent
from modules.vehicle_ai.observability import AgentTraceRecorder
from modules.vehicle_ai.tools import ToolResult
from modules.vehicle_ai.workflow.state import VehicleWorkflowState


ApprovalDecision = Literal["approve", "reject"]


@dataclass(frozen=True)
class WorkflowResult:
    """Stable public view of one stateful workflow checkpoint."""

    thread_id: str
    response: str
    status: str
    reason: str | None
    pending_action: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    interrupted: bool
    interrupt: dict[str, Any] | None
    graph_trace: tuple[str, ...]
    task_id: str = ""
    trace_events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(
            {
                "thread_id": self.thread_id,
                "response": self.response,
                "status": self.status,
                "reason": self.reason,
                "pending_action": self.pending_action,
                "tool_result": self.tool_result,
                "interrupted": self.interrupted,
                "interrupt": self.interrupt,
                "graph_trace": list(self.graph_trace),
                "task_id": self.task_id,
                "trace_events": self.trace_events,
            }
        )


class VehicleAgentWorkflow:
    """Single-cabin LangGraph runtime with explicit approval and recovery routing.

    The existing VehicleAgent remains the bounded model/tool executor and the
    ToolRegistry remains the authorization boundary. LangGraph owns control-flow
    state, checkpointing and human-in-the-loop suspension.

    One workflow instance intentionally binds to one VehicleAgent/session. Create
    another VehicleMindRuntime for an independent cabin session.
    """

    def __init__(
        self,
        agent: VehicleAgent,
        *,
        thread_id: str | None = None,
        checkpointer: Any | None = None,
        trace_recorder: AgentTraceRecorder | None = None,
    ) -> None:
        self.agent = agent
        self.thread_id = thread_id or f"vehiclemind-{uuid4().hex}"
        self.trace_recorder = trace_recorder or AgentTraceRecorder(
            self.thread_id,
            task_id_provider=lambda: self.agent.task.task_id,
        )
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    @property
    def config(self) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": self.thread_id}}

    def _build_graph(self):
        builder = StateGraph(VehicleWorkflowState)
        builder.add_node(
            "user_turn", self._instrument_node("user_turn", self._user_turn)
        )
        builder.add_node(
            "event_turn", self._instrument_node("event_turn", self._event_turn)
        )
        builder.add_node(
            "approval_gate", self._instrument_node("approval_gate", self._approval_gate)
        )
        builder.add_node(
            "execute_approved",
            self._instrument_node("execute_approved", self._execute_approved),
        )
        builder.add_node(
            "reject_action", self._instrument_node("reject_action", self._reject_action)
        )
        builder.add_node(
            "invalid_approval",
            self._instrument_node("invalid_approval", self._invalid_approval),
        )
        builder.add_node(
            "recovery_router",
            self._instrument_node("recovery_router", self._recovery_router),
        )
        builder.add_node("verify", self._instrument_node("verify", self._verify))

        builder.add_conditional_edges(
            START,
            self._route_ingress,
            {"user": "user_turn", "event": "event_turn"},
        )
        builder.add_conditional_edges(
            "user_turn",
            self._route_after_turn,
            {"approval": "approval_gate", "verify": "verify"},
        )
        builder.add_edge("event_turn", "verify")
        builder.add_conditional_edges(
            "approval_gate",
            self._route_approval,
            {
                "approve": "execute_approved",
                "reject": "reject_action",
                "invalid": "invalid_approval",
            },
        )
        builder.add_edge("execute_approved", "recovery_router")
        builder.add_edge("reject_action", "verify")
        builder.add_edge("invalid_approval", "verify")
        builder.add_conditional_edges(
            "recovery_router",
            self._route_recovery,
            {"approval": "approval_gate", "verify": "verify"},
        )
        builder.add_edge("verify", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _instrument_node(self, name: str, handler: Callable) -> Callable:
        def invoke(state: VehicleWorkflowState):
            started = time.perf_counter()
            outcome = "COMPLETED"
            try:
                return handler(state)
            except BaseException as error:
                outcome = (
                    "INTERRUPTED"
                    if type(error).__name__ in {"GraphInterrupt", "GraphBubbleUp"}
                    else "FAILED"
                )
                raise
            finally:
                self.trace_recorder.emit(
                    "graph_node",
                    graph_node=name,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    attributes={"outcome": outcome},
                )

        return invoke

    @staticmethod
    def _append_trace(state: VehicleWorkflowState, node: str) -> list[str]:
        return [*state.get("graph_trace", []), node]

    def _agent_snapshot(self, state: VehicleWorkflowState, node: str) -> dict[str, Any]:
        pending = self.agent.pending_actions.get()
        return {
            "task": self.agent.task.to_dict(),
            "pending_action": pending.to_dict() if pending is not None else None,
            "status": self.agent.task.status.value,
            "reason": self.agent.task.reason,
            "graph_trace": self._append_trace(state, node),
        }

    @staticmethod
    def _route_ingress(state: VehicleWorkflowState) -> str:
        ingress = state.get("ingress")
        if ingress not in {"user", "event"}:
            raise ValueError("workflow ingress must be 'user' or 'event'")
        return ingress

    def _user_turn(self, state: VehicleWorkflowState) -> dict[str, Any]:
        text = str(state.get("user_text", "")).strip()
        if not text:
            raise ValueError("user_text must not be empty")
        response = self.agent.chat(text, debug=bool(state.get("debug", True)))
        return {
            **self._agent_snapshot(state, "user_turn"),
            "response": response,
            "approval": None,
            "tool_result": None,
            "error": None,
        }

    @staticmethod
    def _event_from_dict(raw: dict[str, Any]) -> VehicleEvent:
        return VehicleEvent(
            type=EventType(str(raw["type"])),
            priority=EventPriority(str(raw["priority"])),
            source=str(raw["source"]),
            message=str(raw["message"]),
            data=deepcopy(dict(raw.get("data", {}))),
            timestamp=float(raw["timestamp"]),
            event_id=str(raw["event_id"]),
        )

    def _event_turn(self, state: VehicleWorkflowState) -> dict[str, Any]:
        raw = state.get("event")
        if not isinstance(raw, dict):
            raise ValueError("event ingress requires a serialized VehicleEvent")
        event = self._event_from_dict(raw)
        response = self.agent.recommend_from_event(event)
        return {
            "response": response,
            "status": "EVENT_HANDLED",
            "reason": None,
            "pending_action": None,
            "tool_result": None,
            "approval": None,
            "error": None,
            "graph_trace": self._append_trace(state, "event_turn"),
        }

    @staticmethod
    def _route_after_turn(state: VehicleWorkflowState) -> str:
        return "approval" if state.get("pending_action") else "verify"

    def _approval_gate(self, state: VehicleWorkflowState) -> dict[str, Any]:
        pending = state.get("pending_action")
        if not isinstance(pending, dict):
            return {
                "approval": {"decision": "invalid", "reason": "NO_PENDING_ACTION"},
                "graph_trace": self._append_trace(state, "approval_gate"),
            }

        # Deliberately keep this node side-effect free before interrupt(). LangGraph
        # resumes by restarting the node, so writes belong in execute_approved.
        answer = interrupt(
            {
                "kind": "vehicle_action_approval",
                "question": pending.get("display_text", "确认执行该车机操作？"),
                "action": deepcopy(pending),
                "allowed_decisions": ["approve", "reject"],
            }
        )
        approval = (
            deepcopy(answer)
            if isinstance(answer, dict)
            else {"decision": "invalid", "reason": "MALFORMED_APPROVAL"}
        )
        decision = str(approval.get("decision", "")).casefold()
        if decision not in {"approve", "reject"}:
            decision = "invalid"
        if approval.get("action_id") != pending.get("action_id"):
            decision = "invalid"
            approval["reason"] = "ACTION_ID_MISMATCH"
        approval["decision"] = decision
        return {
            "approval": approval,
            "graph_trace": self._append_trace(state, "approval_gate"),
        }

    @staticmethod
    def _route_approval(state: VehicleWorkflowState) -> str:
        approval = state.get("approval") or {}
        decision = str(approval.get("decision", "invalid"))
        return decision if decision in {"approve", "reject"} else "invalid"

    def _execute_approved(self, state: VehicleWorkflowState) -> dict[str, Any]:
        pending = state.get("pending_action") or {}
        action_id = str(pending.get("action_id", ""))
        live = self.agent.pending_actions.get()
        if live is None or live.action_id != action_id:
            result = ToolResult(
                False,
                "No matching live pending action was found.",
                error="INVALID_CONFIRMATION",
            )
        else:
            result = self.agent.confirm_pending(action_id)
        return {
            **self._agent_snapshot(state, "execute_approved"),
            "response": result.message,
            "tool_result": result.to_dict(),
            "error": result.error,
        }

    def _reject_action(self, state: VehicleWorkflowState) -> dict[str, Any]:
        pending = state.get("pending_action") or {}
        action_id = str(pending.get("action_id", ""))
        result = self.agent.reject_pending(action_id)
        return {
            **self._agent_snapshot(state, "reject_action"),
            "response": "已取消待确认操作。" if result.success else result.message,
            "tool_result": result.to_dict(),
            "error": result.error,
        }

    def _invalid_approval(self, state: VehicleWorkflowState) -> dict[str, Any]:
        return {
            **self._agent_snapshot(state, "invalid_approval"),
            "response": "确认信息无效，敏感操作未执行。",
            "error": "INVALID_APPROVAL",
        }

    def _recovery_router(self, state: VehicleWorkflowState) -> dict[str, Any]:
        # Existing bounded plan recovery may stage a new PendingAction (for example
        # after a simulated POI becomes unavailable). The graph never auto-replays
        # that write: it routes the recovered candidate through approval again.
        tool_result = state.get("tool_result") or {}
        pending = self.agent.pending_actions.get()
        if tool_result.get("error") == "ALTERNATIVE_PENDING":
            plan = self.agent.task.plan
            self.trace_recorder.emit(
                "recovery",
                graph_node="recovery_router",
                attributes={
                    "error": tool_result.get("error"),
                    "pending_action_id": pending.action_id if pending else None,
                    "recovery_count": plan.recovery_count if plan else None,
                },
            )
        return {
            **self._agent_snapshot(state, "recovery_router"),
        }

    @staticmethod
    def _route_recovery(state: VehicleWorkflowState) -> str:
        return "approval" if state.get("pending_action") else "verify"

    def _verify(self, state: VehicleWorkflowState) -> dict[str, Any]:
        if state.get("ingress") == "event":
            return {"graph_trace": self._append_trace(state, "verify")}
        return {
            **self._agent_snapshot(state, "verify"),
        }

    def _result(self) -> WorkflowResult:
        snapshot = self.graph.get_state(self.config)
        values = dict(snapshot.values)
        pending_interrupt = snapshot.interrupts[0] if snapshot.interrupts else None
        interrupt_value = (
            deepcopy(pending_interrupt.value)
            if pending_interrupt is not None
            and isinstance(pending_interrupt.value, dict)
            else None
        )
        ingress = values.get("ingress")
        if pending_interrupt is not None and ingress == "user":
            pending = values.get("pending_action") or {}
            self.trace_recorder.emit(
                "interrupt",
                graph_node="approval_gate",
                attributes={
                    "action_id": pending.get("action_id"),
                    "tool_name": pending.get("tool_name"),
                },
            )
        elif ingress in {"user", "event"}:
            event_type = (
                "task_final_status" if ingress == "user" else "workflow_final_status"
            )
            self.trace_recorder.emit(
                event_type,
                graph_node="verify",
                attributes={
                    "ingress": ingress,
                    "status": values.get("status"),
                    "reason": values.get("reason"),
                },
            )
        return WorkflowResult(
            thread_id=self.thread_id,
            response=str(values.get("response", "")),
            status=str(values.get("status", "")),
            reason=values.get("reason"),
            pending_action=deepcopy(values.get("pending_action")),
            tool_result=deepcopy(values.get("tool_result")),
            interrupted=pending_interrupt is not None,
            interrupt=interrupt_value,
            graph_trace=tuple(values.get("graph_trace", [])),
            task_id=self.agent.task.task_id,
            trace_events=self.trace_recorder.to_dicts(),
        )

    def _ensure_not_waiting(self) -> None:
        snapshot = self.graph.get_state(self.config)
        if snapshot.interrupts:
            self.trace_recorder.emit(
                "ingress_rejected",
                graph_node="approval_gate",
                attributes={"reason": "WAITING_FOR_APPROVAL"},
            )
            raise RuntimeError("workflow is waiting for approval")

    def invoke_user(self, text: str, *, debug: bool = True) -> WorkflowResult:
        self._ensure_not_waiting()
        self.graph.invoke(
            {
                "ingress": "user",
                "user_text": text,
                "debug": debug,
                "event": None,
                "graph_trace": [],
            },
            self.config,
        )
        return self._result()

    def invoke_event(self, event: VehicleEvent) -> WorkflowResult:
        self._ensure_not_waiting()
        self.graph.invoke(
            {
                "ingress": "event",
                "user_text": "",
                "debug": False,
                "event": event.to_dict(),
                "graph_trace": [],
            },
            self.config,
        )
        return self._result()

    def resume(self, decision: ApprovalDecision) -> WorkflowResult:
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be 'approve' or 'reject'")
        snapshot = self.graph.get_state(self.config)
        if not snapshot.interrupts:
            raise RuntimeError("workflow is not waiting for approval")
        pending = snapshot.values.get("pending_action")
        if not isinstance(pending, dict) or not pending.get("action_id"):
            raise RuntimeError("workflow interrupt has no pending action")
        started = time.perf_counter()
        self.trace_recorder.emit(
            "resume",
            graph_node="approval_gate",
            attributes={
                "decision": decision,
                "action_id": str(pending["action_id"]),
            },
        )
        self.graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "action_id": str(pending["action_id"]),
                }
            ),
            self.config,
        )
        self.trace_recorder.emit(
            "resume_completed",
            graph_node="approval_gate",
            latency_ms=(time.perf_counter() - started) * 1000,
            attributes={
                "decision": decision,
                "action_id": str(pending["action_id"]),
            },
        )
        return self._result()

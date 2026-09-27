"""Serializable LangGraph state for the VehicleMind stateful agent runtime."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


IngressKind = Literal["user", "event"]


class VehicleWorkflowState(TypedDict, total=False):
    """Raw workflow state persisted by the LangGraph checkpointer."""

    ingress: IngressKind
    user_text: str
    debug: bool
    event: dict[str, Any] | None
    response: str
    task: dict[str, Any]
    pending_action: dict[str, Any] | None
    approval: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    status: str
    reason: str | None
    error: str | None
    graph_trace: list[str]

"""LangGraph 状态契约：只保存可序列化业务数据，不保存执行授权或客户端。"""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, TypedDict


def append_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """工具轮次追加协议消息；新用户轮次通过 Overwrite 明确替换。"""
    return [*deepcopy(left), *deepcopy(right)]


class AgentGraphState(TypedDict, total=False):
    # 业务状态由节点输出写回；VehicleAgent 属性只是兼容现有工具的视图。
    simulated_vehicle: dict[str, Any]
    task: dict[str, Any]
    pending_action: dict[str, Any] | None
    history: list[dict]
    dialogue_session: dict[str, Any]
    dialogue_proposal: dict[str, Any]
    dialogue_decision: dict[str, Any]
    agent_trace: list[dict]
    trace_sequence: int
    current_user_intent: str
    music_warning: str
    music_paused: bool
    # 每次模型请求使用的上下文及协议消息，与历史事件明确分开。
    messages: Annotated[list[dict], append_messages]
    context: dict[str, Any]
    input_text: str
    debug: bool
    target: str | None
    turn_id: str
    response_text: str
    final_response: str
    # 工具调用逐个经过校验和执行节点，不并行处理车机写操作。
    tool_calls: list[dict]
    call_index: int
    arguments: dict[str, Any]
    tool_result: dict[str, Any] | None
    retry_allowed: bool
    did_retry: bool
    resolution: dict[str, Any] | None
    # 预算随 checkpoint 保存；跨进程恢复时不能复用 monotonic 绝对时间。
    budget: dict[str, Any]
    deadline_at: float
    model_rounds: int
    route: str
    stop_reason: str | None
    # 中断输入只是用户意图，绝不作为可直接执行的确认凭据。
    event: dict[str, Any]
    confirmation_result: dict[str, Any] | None
    graph_trace: list[dict[str, Any]]

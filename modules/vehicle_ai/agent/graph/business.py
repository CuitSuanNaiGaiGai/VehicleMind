"""Graph 状态与既有业务对象的双向转换，保持原任务对象引用可用。"""

from copy import deepcopy
from enum import Enum
from langgraph.types import Overwrite
from dataclasses import asdict, fields
from typing import Any

from modules.vehicle_ai.agent.action_state import PendingAction
from modules.vehicle_ai.agent.dialogue_state import (
    CandidateSnapshot,
    ConstraintValue,
    DialogueSession,
)
from modules.vehicle_ai.agent.plan import TaskPlan, PlanStep, PlanStatus, PlanStepStatus
from modules.vehicle_ai.agent.task_state import AgentTask, TaskStatus


def plain(value):
    """checkpoint 只使用基础数据类型，枚举不依赖反序列化导入。"""
    if isinstance(value, Overwrite):
        return Overwrite(plain(value.value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def capture(agent) -> dict[str, Any]:
    pending = agent.pending_actions.get()
    return plain(
        deepcopy(
            {
                "task": agent.task.to_dict(),
                "pending_action": pending.to_dict() if pending else None,
                "history": agent.history,
                "dialogue_session": asdict(agent.dialogue_session),
                "agent_trace": agent.trace,
                "trace_sequence": agent.trace_sequence,
                "current_user_intent": agent.current_user_intent,
                "music_warning": agent._turn_music_warning,
                "music_paused": agent._turn_music_paused,
            }
        )
    )


def restore(agent, state: dict) -> None:
    raw = deepcopy(state.get("task"))
    if raw is not None:
        raw["status"] = TaskStatus(raw["status"])
        raw["constraints"] = {
            key: ConstraintValue(**value)
            for key, value in raw.get("constraints", {}).items()
        }
        raw["unresolved_constraints"] = [
            ConstraintValue(**value) for value in raw.get("unresolved_constraints", [])
        ]
        plan_data = raw.get("plan")
        if plan_data:
            allowed = {field.name for field in fields(TaskPlan)}
            plan_data = {
                key: value for key, value in plan_data.items() if key in allowed
            }
            plan_data["status"] = PlanStatus(plan_data["status"])
            plan_data["steps"] = [
                PlanStep(**{**step, "status": PlanStepStatus(step["status"])})
                for step in plan_data["steps"]
            ]
            snapshot = plan_data.get("candidate_snapshot")
            if snapshot:
                snapshot["presented_poi_ids"] = tuple(snapshot["presented_poi_ids"])
                plan_data["candidate_snapshot"] = CandidateSnapshot(**snapshot)
            existing = agent.task.plan
            if existing is not None and agent.task.task_id == raw["task_id"]:
                for key, value in plan_data.items():
                    setattr(existing, key, value)
                raw["plan"] = existing
            else:
                raw["plan"] = TaskPlan(**plan_data)
        if agent.task.task_id == raw["task_id"]:
            for key, value in raw.items():
                setattr(agent.task, key, value)
        else:
            agent.task = AgentTask(**raw)
    if "pending_action" in state:
        pending = state["pending_action"]
        if pending:
            agent.pending_actions.set(PendingAction(**deepcopy(pending)))
        else:
            agent.pending_actions.clear()
    agent.history = deepcopy(state.get("history", agent.history))
    agent.trace = deepcopy(state.get("agent_trace", agent.trace))
    agent.trace_sequence = state.get("trace_sequence", agent.trace_sequence)
    agent.current_user_intent = state.get("current_user_intent", "")
    agent._turn_music_warning = state.get("music_warning", "")
    agent._turn_music_paused = state.get("music_paused", False)
    if state.get("dialogue_session"):
        agent.dialogue_session = DialogueSession(**state["dialogue_session"])

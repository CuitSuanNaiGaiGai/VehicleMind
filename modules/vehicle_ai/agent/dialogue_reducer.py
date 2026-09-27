"""Pure reduction of a validated dialogue proposal into a task decision."""

from dataclasses import dataclass

from modules.vehicle_ai.agent.dialogue_state import ConstraintValue
from modules.vehicle_ai.agent.task_state import AgentTask, TaskStatus


@dataclass(frozen=True)
class DialogueDecision:
    operation: str
    reason: str | None
    goal: str
    constraints: dict[str, ConstraintValue]
    unresolved_constraints: list[ConstraintValue]
    revision: int
    reference: dict | None
    clarification: str | None
    changed: bool = False


_TERMINAL = {TaskStatus.IDLE, TaskStatus.CANCELLED, TaskStatus.COMPLETED, TaskStatus.FAILED}


def _copy_state(task: AgentTask) -> tuple[dict, list]:
    return dict(task.constraints), list(task.unresolved_constraints)


def _conflicting_changes(intent: str, changes: list[dict]) -> bool:
    if intent in {"SELECT", "ASK_CANDIDATE"} and changes:
        return True
    by_field: dict[str, list[dict]] = {}
    for change in changes:
        by_field.setdefault(change["field"], []).append(change)
    for field, group in by_field.items():
        signatures = {(item["op"], item["value"]) for item in group}
        if field != "unsupported" and len(signatures) > 1:
            return True
        if field == "unsupported" and any(
            (item["op"] == "REMOVE" and item["value"] == "*") for item in group
        ) and len(group) > 1:
            return True
        if field == "unsupported" and any(
            item["op"] == "SET" and ("REMOVE", item["value"]) in signatures
            for item in group
        ):
            return True
    return False


def _apply_changes(constraints: dict, unresolved: list, changes: list[dict], turn_id: int) -> bool:
    changed = False
    for item in changes:
        field, op, value = item["field"], item["op"], item["value"]
        if field in {"max_distance_km", "preferred_area"}:
            if op == "REMOVE":
                if field in constraints:
                    del constraints[field]
                    changed = True
            elif field not in constraints or constraints[field].value != value:
                stored_value = float(value) if field == "max_distance_km" else value
                constraints[field] = ConstraintValue(stored_value, turn_id, item["evidence"])
                changed = True
        elif op == "SET":
            if not any(current.value == value for current in unresolved):
                unresolved.append(ConstraintValue(value, turn_id, item["evidence"]))
                changed = True
        elif value == "*":
            if unresolved:
                unresolved.clear()
                changed = True
        else:
            remaining = [current for current in unresolved if current.value != value]
            if len(remaining) != len(unresolved):
                unresolved[:] = remaining
                changed = True
    return changed


def _message(reason: str, unresolved: list[ConstraintValue], proposal: dict) -> str:
    if reason == "UNSUPPORTED_CONSTRAINT":
        values = "、".join(str(item.value) for item in unresolved)
        return f"暂不支持这些条件：{values}。可按已支持条件继续，或明确放弃它们。"
    if reason == "NEW_TASK_REQUIRED":
        return "当前任务已结束；如需继续，请明确说明新的查找目标。"
    if reason == "REVISION_LIMIT":
        return "本任务已达到修改次数上限；请明确说明新的查找目标。"
    if reason == "CONFLICTING_CONSTRAINTS":
        return "这句话包含相互冲突的选择或条件，请明确要保留哪一项。"
    return proposal.get("clarification_reason") or "我不确定你的意思，请再说明一次。"


def _decision(task: AgentTask, operation: str, reason: str | None, goal: str,
              constraints: dict, unresolved: list, revision: int, reference: dict | None,
              clarification: str | None, changed: bool = False) -> DialogueDecision:
    return DialogueDecision(
        operation, reason, goal, dict(constraints), list(unresolved), revision,
        dict(reference) if reference is not None else None, clarification, changed
    )


def _clarify(task: AgentTask, reason: str, proposal: dict, *, goal: str | None = None,
             constraints: dict | None = None, unresolved: list | None = None,
             revision: int | None = None, changed: bool = False) -> DialogueDecision:
    current_constraints, current_unresolved = _copy_state(task)
    constraints = current_constraints if constraints is None else constraints
    unresolved = current_unresolved if unresolved is None else unresolved
    message = _message(reason, unresolved, proposal)
    return _decision(
        task, "clarify", reason, task.goal if goal is None else goal, constraints,
        unresolved, task.revision if revision is None else revision, None, message, changed
    )


def reduce_dialogue(task: AgentTask, proposal: dict, *, turn_id: int,
                    max_revisions: int = 3) -> DialogueDecision:
    """Return a proposed task state without mutating the input task."""
    intent = proposal["intent"]
    changes = proposal["changes"]
    if _conflicting_changes(intent, changes):
        return _clarify(task, "CONFLICTING_CONSTRAINTS", proposal)
    if intent == "UNCLEAR":
        return _clarify(task, "AMBIGUOUS_REQUEST", proposal)
    if intent == "SIDE_QUESTION":
        constraints, unresolved = _copy_state(task)
        return _decision(task, "side_question", None, task.goal, constraints, unresolved,
                         task.revision, None, None)
    if intent == "CANCEL":
        constraints, unresolved = _copy_state(task)
        return _decision(task, "cancel", None, task.goal, constraints, unresolved,
                         task.revision, None, None)
    if task.status in _TERMINAL and intent in {"UPDATE_CONSTRAINTS", "RESUME", "SELECT"}:
        return _clarify(task, "NEW_TASK_REQUIRED", proposal)
    if intent == "RESUME":
        constraints, unresolved = _copy_state(task)
        if unresolved:
            return _clarify(task, "UNSUPPORTED_CONSTRAINT", proposal)
        return _decision(task, "resume", None, task.goal, constraints, unresolved,
                         task.revision, None, None)
    if intent in {"SELECT", "ASK_CANDIDATE"}:
        constraints, unresolved = _copy_state(task)
        if unresolved:
            return _clarify(task, "UNSUPPORTED_CONSTRAINT", proposal)
        operation = "select" if intent == "SELECT" else "candidate_answer"
        return _decision(task, operation, None, task.goal, constraints, unresolved,
                         task.revision, proposal["reference"], None)
    if intent not in {"START", "UPDATE_CONSTRAINTS"}:
        return _clarify(task, "AMBIGUOUS_REQUEST", proposal)

    is_new = intent == "START" and task.status in _TERMINAL
    goal = proposal["evidence"] if intent == "START" else task.goal
    if is_new:
        constraints, unresolved = {}, []
        changed = True
    else:
        constraints, unresolved = _copy_state(task)
        changed = goal != task.goal
    changed = _apply_changes(constraints, unresolved, changes, turn_id) or changed
    if not is_new and changed and task.revision >= max_revisions:
        return _clarify(task, "REVISION_LIMIT", proposal)
    revision = 0 if is_new else task.revision + int(changed)
    operation = "start" if is_new else ("search" if changed else "resume")
    if unresolved:
        return _clarify(task, "UNSUPPORTED_CONSTRAINT", proposal, goal=goal,
                        constraints=constraints, unresolved=unresolved, revision=revision,
                        changed=changed)
    return _decision(task, operation, None, goal, constraints, unresolved, revision,
                     None, None, changed)

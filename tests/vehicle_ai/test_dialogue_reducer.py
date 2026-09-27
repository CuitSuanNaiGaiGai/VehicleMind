from copy import deepcopy

from modules.vehicle_ai.agent.dialogue_reducer import reduce_dialogue
from modules.vehicle_ai.agent.dialogue_state import ConstraintValue
from modules.vehicle_ai.agent.task_state import AgentTask, TaskStatus


def proposal(intent, changes=(), reference=None, evidence="", reason=None):
    return {
        "intent": intent,
        "changes": list(changes),
        "reference": reference,
        "evidence": evidence,
        "clarification_reason": reason,
    }


def change(field, value, evidence, op="SET"):
    return {"field": field, "op": op, "value": value, "evidence": evidence}


def test_update_and_remove_are_explicit_and_source_checked():
    task = AgentTask(goal="找附近服务区", status=TaskStatus.AWAITING_INPUT)
    task.constraints["preferred_area"] = ConstraintValue("西湖", 1, "西湖附近")
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 15.0, "十五公里以内")],
        evidence="十五公里以内",
    )
    before = deepcopy(task.to_dict())

    decision = reduce_dialogue(task, update, turn_id=2)

    assert task.to_dict() == before
    assert decision.revision == 1 and decision.operation == "search"
    assert decision.constraints["preferred_area"].value == "西湖"
    assert decision.constraints["max_distance_km"].source_turn_id == 2


def test_omitted_supported_constraints_are_retained():
    task = AgentTask(status=TaskStatus.RUNNING)
    task.constraints = {
        "preferred_area": ConstraintValue("西湖", 1, "西湖附近"),
        "max_distance_km": ConstraintValue(15.0, 1, "十五公里内"),
    }
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 10.0, "十公里以内")],
    )

    decision = reduce_dialogue(task, update, turn_id=2)

    assert decision.constraints["preferred_area"] == task.constraints["preferred_area"]
    assert decision.constraints["max_distance_km"].value == 10.0
    assert decision.revision == 1


def test_explicit_remove_deletes_only_the_named_supported_constraint():
    task = AgentTask(status=TaskStatus.AWAITING_INPUT)
    task.constraints = {
        "preferred_area": ConstraintValue("西湖", 1, "西湖附近"),
        "max_distance_km": ConstraintValue(15.0, 1, "十五公里内"),
    }
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("preferred_area", None, "取消区域偏好", "REMOVE")],
    )

    decision = reduce_dialogue(task, update, turn_id=2)

    assert "preferred_area" not in decision.constraints
    assert decision.constraints["max_distance_km"].value == 15.0
    assert decision.revision == 1


def test_removing_a_missing_supported_constraint_is_a_noop():
    task = AgentTask(status=TaskStatus.RUNNING, revision=1)
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("preferred_area", None, "取消区域偏好", "REMOVE")],
    )

    decision = reduce_dialogue(task, update, turn_id=2)

    assert decision.changed is False
    assert decision.operation == "resume"
    assert decision.revision == 1
    assert decision.constraints == {}


def test_unsupported_condition_blocks_search_but_supported_changes_commit():
    task = AgentTask(goal="找服务区", status=TaskStatus.RUNNING)
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [
            change("max_distance_km", 15.0, "15公里以内"),
            change("unsupported", "车载按摩椅", "车载按摩椅"),
        ],
    )

    decision = reduce_dialogue(task, update, turn_id=2)

    assert decision.operation == "clarify"
    assert decision.reason == "UNSUPPORTED_CONSTRAINT"
    assert "车载按摩椅" in decision.clarification
    assert decision.constraints["max_distance_km"].value == 15.0
    assert decision.unresolved_constraints == [
        ConstraintValue("车载按摩椅", 2, "车载按摩椅")
    ]
    assert decision.revision == 1
    assert task.constraints == {} and task.unresolved_constraints == []


def test_explicitly_abandoning_all_unsupported_conditions_clears_them():
    task = AgentTask(status=TaskStatus.AWAITING_INPUT)
    task.unresolved_constraints = [
        ConstraintValue("车载按摩椅", 1, "要按摩椅"),
        ConstraintValue("车载冰箱", 1, "要车载冰箱"),
    ]
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("unsupported", "*", "按已支持条件继续，放弃其他要求", "REMOVE")],
    )

    decision = reduce_dialogue(task, update, turn_id=2)

    assert decision.unresolved_constraints == []
    assert decision.reason is None
    assert decision.operation == "search"
    assert decision.revision == 1


def test_releasing_one_unsupported_condition_retains_the_others():
    task = AgentTask(status=TaskStatus.RUNNING)
    task.unresolved_constraints = [
        ConstraintValue("按摩椅", 1, "想要按摩椅"),
        ConstraintValue("车载冰箱", 1, "想要车载冰箱"),
    ]
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("unsupported", "按摩椅", "放弃按摩椅", "REMOVE")],
    )

    decision = reduce_dialogue(task, update, turn_id=2)

    assert [item.value for item in decision.unresolved_constraints] == ["车载冰箱"]
    assert decision.reason == "UNSUPPORTED_CONSTRAINT"
    assert decision.operation == "clarify"


def test_fourth_real_revision_is_rejected_without_changing_old_conditions():
    task = AgentTask(status=TaskStatus.RUNNING, revision=3)
    task.constraints["max_distance_km"] = ConstraintValue(15.0, 3, "十五公里内")
    before = deepcopy(task.to_dict())
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 10.0, "十公里以内")],
    )

    decision = reduce_dialogue(task, update, turn_id=4)

    assert decision.reason == "REVISION_LIMIT"
    assert decision.operation == "clarify"
    assert decision.constraints == task.constraints
    assert decision.revision == 3
    assert task.to_dict() == before


def test_setting_an_existing_value_is_not_a_real_revision():
    task = AgentTask(status=TaskStatus.RUNNING, revision=2)
    task.constraints["max_distance_km"] = ConstraintValue(15.0, 1, "十五公里以内")
    update = proposal(
        "UPDATE_CONSTRAINTS",
        [change("max_distance_km", 15.0, "15公里以内")],
    )

    decision = reduce_dialogue(task, update, turn_id=3)

    assert decision.changed is False
    assert decision.operation == "resume"
    assert decision.revision == 2
    assert decision.constraints["max_distance_km"].source_turn_id == 1


def test_new_start_does_not_inherit_conditions_from_a_terminal_task():
    task = AgentTask(goal="旧目标", status=TaskStatus.COMPLETED, revision=3)
    task.constraints["max_distance_km"] = ConstraintValue(15.0, 2, "旧距离")
    start = proposal(
        "START",
        [change("preferred_area", "西湖", "西湖附近")],
        evidence="找西湖附近的服务区",
    )

    decision = reduce_dialogue(task, start, turn_id=5)

    assert decision.operation == "start"
    assert decision.goal == "找西湖附近的服务区"
    assert decision.revision == 0
    assert set(decision.constraints) == {"preferred_area"}
    assert "max_distance_km" not in decision.constraints


def test_start_during_active_work_is_a_revision_not_a_new_task():
    task = AgentTask(goal="找服务区", status=TaskStatus.RUNNING)
    start = proposal(
        "START",
        [change("preferred_area", "西湖", "西湖附近")],
        evidence="找西湖附近的服务区",
    )

    decision = reduce_dialogue(task, start, turn_id=2)

    assert decision.operation == "search"
    assert decision.goal == "找西湖附近的服务区"
    assert decision.revision == 1
    assert task.task_id and task.goal == "找服务区"


def test_updates_resume_and_selection_cannot_revive_terminal_tasks():
    cases = [
        proposal("UPDATE_CONSTRAINTS", [change("max_distance_km", 10.0, "十公里")]),
        proposal("RESUME"),
        proposal("SELECT", reference={"index": 1, "name": None, "evidence": "第1个"}),
    ]
    for item in cases:
        task = AgentTask(status=TaskStatus.COMPLETED)
        decision = reduce_dialogue(task, item, turn_id=2)
        assert decision.operation == "clarify"
        assert decision.reason == "NEW_TASK_REQUIRED"


def test_side_question_preserves_business_state():
    task = AgentTask(
        goal="找服务区", status=TaskStatus.AWAITING_CONFIRMATION, revision=2
    )
    task.constraints["preferred_area"] = ConstraintValue("西湖", 1, "西湖附近")
    task.pending_action = {"action_id": "opaque-action", "kind": "navigation"}
    before = deepcopy(task.to_dict())

    decision = reduce_dialogue(task, proposal("SIDE_QUESTION"), turn_id=3)

    assert decision.operation == "side_question"
    assert decision.constraints == task.constraints
    assert decision.changed is False
    assert task.to_dict() == before


def test_unclear_preserves_a_pending_confirmation_and_task_state():
    task = AgentTask(goal="找服务区", status=TaskStatus.AWAITING_CONFIRMATION)
    task.pending_action = {"action_id": "opaque-action", "kind": "navigation"}
    task.clarification = "之前的问题"
    before = deepcopy(task.to_dict())

    decision = reduce_dialogue(
        task,
        proposal("UNCLEAR", evidence="好", reason="指代不明确"),
        turn_id=2,
    )

    assert decision.operation == "clarify"
    assert decision.reason == "AMBIGUOUS_REQUEST"
    assert decision.clarification == "指代不明确"
    assert task.to_dict() == before


def test_conflicting_changes_and_selection_with_changes_are_rejected_as_a_group():
    conflicted = proposal(
        "UPDATE_CONSTRAINTS",
        [
            change("max_distance_km", 15.0, "15公里"),
            change("max_distance_km", 10.0, "10公里"),
        ],
    )
    mixed = proposal(
        "SELECT",
        [change("max_distance_km", 15.0, "15公里")],
        reference={"index": 1, "name": None, "evidence": "选第1个"},
    )
    for item in (conflicted, mixed):
        task = AgentTask(status=TaskStatus.RUNNING)
        decision = reduce_dialogue(task, item, turn_id=2)
        assert decision.operation == "clarify"
        assert decision.reason == "CONFLICTING_CONSTRAINTS"
        assert decision.constraints == {}
        assert decision.revision == 0


def test_selection_and_candidate_question_preserve_user_reference():
    for intent, operation in (
        ("SELECT", "select"),
        ("ASK_CANDIDATE", "candidate_answer"),
    ):
        reference = {"index": 2, "name": None, "evidence": "第2个"}
        task = AgentTask(status=TaskStatus.RUNNING)
        decision = reduce_dialogue(
            task,
            proposal(intent, reference=reference),
            turn_id=2,
        )
        assert decision.operation == operation
        assert decision.reference == reference
        assert decision.reference is not reference
        assert decision.revision == 0


def test_cancel_and_resume_only_return_operations_without_mutating_task():
    for intent, operation in (("CANCEL", "cancel"), ("RESUME", "resume")):
        task = AgentTask(status=TaskStatus.AWAITING_INPUT, revision=2)
        before = deepcopy(task.to_dict())
        decision = reduce_dialogue(task, proposal(intent), turn_id=3)
        assert decision.operation == operation
        assert task.to_dict() == before

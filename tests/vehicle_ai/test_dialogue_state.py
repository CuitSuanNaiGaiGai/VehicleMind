import pytest

from modules.config.agent_dialogue import AgentDialogueConfig
from modules.vehicle_ai.agent.budget import (
    AgentBudgetConfig,
    BudgetExceeded,
    TurnBudget,
)
from modules.vehicle_ai.agent.dialogue_state import (
    CandidateSnapshot,
    ConstraintValue,
    DialogueSession,
)
from modules.vehicle_ai.agent.plan import TaskPlan
from modules.vehicle_ai.agent.task_state import AgentTask


def test_dialogue_state_isolated_and_budget_compatible() -> None:
    assert AgentDialogueConfig.load().enabled is False
    assert AgentBudgetConfig.load().max_tool_rounds == 5
    left, right = AgentTask(), AgentTask()
    left.constraints["max_distance_km"] = ConstraintValue(15.0, 2, "十五公里以内")
    assert right.constraints == {}
    assert left.unresolved_constraints is not right.unresolved_constraints

    snapshot = left.to_dict()
    snapshot["constraints"]["max_distance_km"]["value"] = 1.0
    assert left.constraints["max_distance_km"].value == 15.0

    session = DialogueSession()
    assert session.next_turn() == 1
    assert session.next_turn() == 2

    now = [0.0]
    budget = TurnBudget(90, 10, lambda: now[0], max_model_calls=5)
    for _ in range(5):
        assert budget.claim_model() == 90
    with pytest.raises(BudgetExceeded, match="MODEL_BUDGET"):
        budget.claim_model()
    assert budget.model_calls == 5
    now[0] = 90.0
    with pytest.raises(BudgetExceeded, match="TIME_BUDGET"):
        budget.remaining()


def test_dialogue_session_ids_are_unique_and_turn_id_is_session_local() -> None:
    first, second = DialogueSession(), DialogueSession()

    assert first.session_id != second.session_id
    assert first.turn_id == second.turn_id == 0
    assert first.next_turn() == 1
    assert second.turn_id == 0


def test_task_plan_serializes_candidate_snapshot_as_independent_metadata() -> None:
    candidate_snapshot = CandidateSnapshot(
        candidate_set_id="set-1",
        constraint_revision=2,
        created_at=10.5,
        presented_turn_id=3,
        expires_at=130.5,
        presented_poi_ids=("poi-1", "poi-2"),
    )
    plan = TaskPlan(
        goal="找服务区",
        max_steps=3,
        max_recoveries=1,
        candidate_snapshot=candidate_snapshot,
    )

    result = plan.to_dict()
    assert result["candidate_snapshot"] == {
        "candidate_set_id": "set-1",
        "constraint_revision": 2,
        "created_at": 10.5,
        "presented_turn_id": 3,
        "expires_at": 130.5,
        "presented_poi_ids": ("poi-1", "poi-2"),
    }
    result["candidate_snapshot"]["presented_turn_id"] = 9
    assert plan.candidate_snapshot.presented_turn_id == 3
    assert "candidates" not in result["candidate_snapshot"]
    assert (
        TaskPlan(goal="旧调用", max_steps=1, max_recoveries=0).candidate_snapshot
        is None
    )


def test_task_state_defaults_and_serializes_new_dialogue_fields() -> None:
    first, second = AgentTask(), AgentTask()
    first.constraints["max_distance_km"] = ConstraintValue(15.0, 1, "十五公里以内")
    first.unresolved_constraints.append(ConstraintValue("休息", 2, "想休息一下"))
    first.clarification = "更偏好哪种设施？"
    first.revision = 2
    first.last_presented_candidate_set_id = "set-1"

    result = first.to_dict()
    assert result["revision"] == 2
    assert result["constraints"]["max_distance_km"] == {
        "value": 15.0,
        "source_turn_id": 1,
        "evidence": "十五公里以内",
    }
    assert result["unresolved_constraints"][0]["value"] == "休息"
    assert result["clarification"] == "更偏好哪种设施？"
    assert result["last_presented_candidate_set_id"] == "set-1"
    assert second.constraints == {}
    assert second.unresolved_constraints == []


def test_constraint_value_and_candidate_snapshot_are_immutable() -> None:
    constraint = ConstraintValue(12.0, 1, "十二公里内")
    snapshot = CandidateSnapshot("set-1", 1, 0.0, None, 120.0, ())

    with pytest.raises((AttributeError, TypeError)):
        constraint.value = 13.0
    with pytest.raises((AttributeError, TypeError)):
        snapshot.expires_at = 1.0

from __future__ import annotations

import pytest

from modules.vehicle_ai.agent.plan import (
    PlanStatus,
    PlanStepStatus,
    PlanTransitionError,
    TaskPlan,
)


def test_task_plan_records_step_evidence_and_remaining_budget() -> None:
    plan = TaskPlan(goal="导航到附近服务区", max_steps=3, max_recoveries=1)

    plan.start_step("SEARCH", success_condition="返回规范 POI 候选", at=10.0)
    plan.finish_step(
        "SEARCH",
        success=True,
        evidence_summary="找到 2 个模拟候选",
        at=10.5,
    )

    result = plan.to_dict()
    assert result["status"] == PlanStatus.RUNNING.value
    assert result["steps"][0] == {
        "name": "SEARCH",
        "status": PlanStepStatus.SUCCEEDED.value,
        "started_at": 10.0,
        "finished_at": 10.5,
        "success_condition": "返回规范 POI 候选",
        "evidence_summary": "找到 2 个模拟候选",
        "error_code": None,
    }
    assert result["remaining_step_budget"] == 2


def test_step_budget_overflow_stops_plan_without_recording_extra_step() -> None:
    plan = TaskPlan(goal="休息", max_steps=1, max_recoveries=1)
    plan.start_step("SEARCH", success_condition="存在候选", at=1.0)
    plan.finish_step("SEARCH", success=False, error_code="NO_RESULTS", at=2.0)

    with pytest.raises(PlanTransitionError, match="STEP_BUDGET_EXCEEDED"):
        plan.start_step("SELECT", success_condition="选中候选", at=3.0)

    assert plan.status is PlanStatus.FAILED
    assert plan.terminal_reason == "STEP_BUDGET_EXCEEDED"
    assert len(plan.steps) == 1


def test_plan_allows_only_one_recovery_and_preserves_terminal_state() -> None:
    plan = TaskPlan(goal="休息", max_steps=9, max_recoveries=1)

    assert plan.claim_recovery("POI_UNAVAILABLE") is True
    assert plan.claim_recovery("POI_UNAVAILABLE") is False
    assert plan.status is PlanStatus.FAILED
    assert plan.terminal_reason == "RECOVERY_BUDGET_EXCEEDED"


def test_recovery_reason_is_auditable_without_marking_plan_terminal() -> None:
    plan = TaskPlan(goal="休息", max_steps=9, max_recoveries=1)

    assert plan.claim_recovery("POI_UNAVAILABLE") is True
    snapshot = plan.to_dict()

    assert snapshot["max_steps"] == 9
    assert snapshot["max_recoveries"] == 1
    assert snapshot["recovery_reasons"] == ["POI_UNAVAILABLE"]
    assert snapshot["terminal_reason"] is None


def test_plan_cannot_finish_a_step_that_is_not_running() -> None:
    plan = TaskPlan(goal="休息", max_steps=3, max_recoveries=1)

    with pytest.raises(PlanTransitionError, match="STEP_NOT_RUNNING"):
        plan.finish_step("SEARCH", success=True, at=2.0)


def test_completed_plan_cannot_accept_more_steps() -> None:
    plan = TaskPlan(goal="休息", max_steps=3, max_recoveries=1)
    plan.start_step("VERIFY", success_condition="导航状态为 ACTIVE", at=1.0)
    plan.finish_step("VERIFY", success=True, at=2.0)
    plan.complete(at=2.0)

    with pytest.raises(PlanTransitionError, match="PLAN_TERMINAL"):
        plan.start_step("SEARCH", success_condition="候选存在", at=3.0)

    assert plan.status is PlanStatus.COMPLETED


@pytest.mark.parametrize(
    ("goal", "max_steps", "max_recoveries"),
    [(" ", 1, 0), ("休息", 0, 0), ("休息", 1, -1)],
)
def test_plan_rejects_empty_goal_and_invalid_budgets(
    goal: str, max_steps: int, max_recoveries: int
) -> None:
    with pytest.raises(ValueError):
        TaskPlan(goal=goal, max_steps=max_steps, max_recoveries=max_recoveries)


def test_plan_rejects_overlapping_and_mismatched_step_transitions() -> None:
    plan = TaskPlan(goal="休息", max_steps=3, max_recoveries=1)
    plan.start_step("SEARCH", success_condition="返回候选", at=1.0)

    with pytest.raises(PlanTransitionError, match="STEP_ALREADY_RUNNING"):
        plan.start_step("SELECT", success_condition="选择候选", at=2.0)
    with pytest.raises(PlanTransitionError, match="STEP_NOT_RUNNING"):
        plan.finish_step("SELECT", success=True, at=2.0)

    plan.finish_step("SEARCH", success=True, at=3.0)
    with pytest.raises(PlanTransitionError, match="STEP_NOT_RUNNING"):
        plan.finish_step("SEARCH", success=True, at=4.0)


def test_skipped_step_and_terminal_transitions_preserve_plan_state() -> None:
    plan = TaskPlan(goal="休息", max_steps=3, max_recoveries=1)
    skipped = plan.skip_step(
        "KNOWLEDGE",
        reason="当前问题无需知识检索",
        success_condition="可跳过知识检索",
        at=1.0,
    )
    assert skipped.status is PlanStepStatus.SKIPPED
    assert skipped.evidence_summary == "当前问题无需知识检索"

    plan.start_step("SEARCH", success_condition="返回候选", at=2.0)
    with pytest.raises(PlanTransitionError, match="STEP_STILL_RUNNING"):
        plan.complete()
    with pytest.raises(ValueError, match="stop requires"):
        plan.stop(PlanStatus.COMPLETED, "not allowed")

    plan.finish_step("SEARCH", success=False, error_code="NO_RESULTS", at=3.0)
    plan.stop(PlanStatus.CANCELLED, "USER_CANCELLED")
    plan.stop(PlanStatus.FAILED, "later stop must not overwrite terminal state")
    assert plan.status is PlanStatus.CANCELLED
    assert plan.terminal_reason == "USER_CANCELLED"


def test_exhausted_recovery_marks_active_step_failed() -> None:
    plan = TaskPlan(goal="休息", max_steps=2, max_recoveries=0)
    step = plan.start_step("SEARCH", success_condition="返回候选", at=1.0)

    assert plan.claim_recovery("TRANSIENT_SEARCH_FAILURE") is False
    assert step.status is PlanStepStatus.FAILED
    assert step.error_code == "RECOVERY_BUDGET_EXCEEDED"
    assert plan.status is PlanStatus.FAILED

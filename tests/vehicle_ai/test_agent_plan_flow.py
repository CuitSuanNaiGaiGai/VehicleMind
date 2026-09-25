from __future__ import annotations

from modules.vehicle_ai.agent.plan import PlanStatus
from modules.config.agent_plan import AgentPlanConfig
from modules.vehicle_ai.context import NavigationState
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import NavigationConfig
from modules.vehicle_ai.tools.base import ToolResult


def _search_call(max_distance_km: float | None = None) -> ScriptedResponse:
    arguments = {} if max_distance_km is None else {"max_distance_km": max_distance_km}
    import json

    return ScriptedResponse(
        content=None,
        tool_calls=(
            LLMToolCall(
                "search-1",
                "search_nearby_rest_area",
                arguments,
                json.dumps(arguments),
            ),
        ),
    )


def _navigation_call(poi_id: str) -> ScriptedResponse:
    import json

    arguments = {"poi_id": poi_id}
    return ScriptedResponse(
        content=None,
        tool_calls=(
            LLMToolCall(
                f"navigate-{poi_id}",
                "start_navigation",
                arguments,
                json.dumps(arguments),
            ),
        ),
    )


def _catalog() -> tuple[dict, ...]:
    return (
        {
            "poi_id": "rest_area_001",
            "name": "西湖服务区",
            "aliases": ["西湖休息区"],
            "distance_km": 6.8,
            "eta_minutes": 8,
        },
        {
            "poi_id": "rest_area_002",
            "name": "河滨服务区",
            "aliases": ["河滨休息区"],
            "distance_km": 12.4,
            "eta_minutes": 15,
        },
    )


def _runtime(
    *responses,
    unavailable_poi_ids: frozenset[str] = frozenset(),
    transient_search_failures: int = 0,
) -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient(responses),
        navigation_config=NavigationConfig(
            poi_catalog=_catalog(),
            unavailable_poi_ids=unavailable_poi_ids,
            transient_search_failures=transient_search_failures,
        ),
    )


def test_agent_stages_the_model_selected_canonical_candidate() -> None:
    runtime = _runtime(_search_call(), _navigation_call("rest_area_002"))

    runtime.chat("帮我找附近服务区并导航", debug=False)

    pending = runtime.agent.pending_actions.get()
    assert pending is not None
    assert pending.arguments["poi_id"] == "rest_area_002"
    plan = runtime.agent.task.to_dict()["plan"]
    assert plan["selected_poi_id"] == "rest_area_002"
    assert [step["name"] for step in plan["steps"]][-2:] == [
        "SELECT",
        "AWAIT_CONFIRMATION",
    ]
    assert (
        runtime.agent.plan_flow.before_tool(
            runtime.agent,
            "start_navigation",
            {"poi_id": "rest_area_002"},
        )
        is None
    )


def test_plan_stops_when_search_has_no_feasible_candidate() -> None:
    runtime = _runtime(
        _search_call(max_distance_km=1.0),
        ScriptedResponse(content="没有找到符合条件的休息地点。"),
    )

    runtime.chat("找 1 公里内的休息区", debug=False)

    assert (
        runtime.agent.task.to_dict()["plan"]["status"] == PlanStatus.STOPPED_NO_RESULT
    )
    assert runtime.agent.task.to_dict()["plan"]["terminal_reason"] == "NO_RESULTS"
    assert runtime.agent.pending_actions.get() is None


def test_new_unrelated_request_closes_a_running_search_plan() -> None:
    runtime = _runtime(
        _search_call(),
        ScriptedResponse(content="找到西湖服务区。"),
        ScriptedResponse(content="空调已打开。"),
    )

    runtime.chat("帮我找附近服务区", debug=False)
    assert runtime.agent.task.to_dict()["plan"]["status"] == PlanStatus.RUNNING

    runtime.chat("打开空调。", debug=False)

    stopped = [
        event
        for event in runtime.agent.trace
        if event["kind"] == "plan_step" and event["event"] == "PLAN_STOPPED"
    ]
    assert stopped[-1]["plan"]["status"] == PlanStatus.CANCELLED
    assert stopped[-1]["plan"]["terminal_reason"] == "NEW_TASK"
    assert runtime.agent.task.to_dict()["plan"] is None


def test_unavailable_primary_stages_alternative_with_new_confirmation() -> None:
    runtime = _runtime(
        _search_call(),
        _navigation_call("rest_area_001"),
        unavailable_poi_ids=frozenset({"rest_area_001"}),
    )
    runtime.chat("帮我找附近服务区并导航", debug=False)
    original = runtime.agent.pending_actions.get()
    assert original is not None

    failed = runtime.agent.confirm_pending(original.action_id)

    replacement = runtime.agent.pending_actions.get()
    assert failed.success is False
    assert failed.error == "ALTERNATIVE_PENDING"
    assert replacement is not None
    assert replacement.action_id != original.action_id
    assert replacement.arguments["poi_id"] == "rest_area_002"
    assert (
        runtime.context_manager.get_context().vehicle.navigation_state
        == NavigationState.IDLE
    )
    assert runtime.agent.task.status.value == "AWAITING_CONFIRMATION"
    assert runtime.agent.task.to_dict()["plan"]["recovery_count"] == 1

    completed = runtime.agent.confirm_pending(replacement.action_id)

    assert completed.success is True
    assert runtime.agent.task.to_dict()["plan"]["status"] == PlanStatus.COMPLETED
    assert (
        runtime.context_manager.get_context().vehicle.navigation_destination_id
        == "rest_area_002"
    )


def test_unavailable_primary_cannot_be_recovered_twice() -> None:
    runtime = _runtime(
        _search_call(),
        _navigation_call("rest_area_001"),
        unavailable_poi_ids=frozenset({"rest_area_001", "rest_area_002"}),
    )
    runtime.chat("帮我找附近服务区并导航", debug=False)
    original = runtime.agent.pending_actions.get()
    assert original is not None
    first_failure = runtime.agent.confirm_pending(original.action_id)
    replacement = runtime.agent.pending_actions.get()
    assert replacement is not None
    second_failure = runtime.agent.confirm_pending(replacement.action_id)

    assert first_failure.error == "ALTERNATIVE_PENDING"
    assert second_failure.error == "POI_UNAVAILABLE"
    assert runtime.agent.pending_actions.get() is None
    assert runtime.agent.task.to_dict()["plan"]["status"] == PlanStatus.FAILED
    assert (
        runtime.agent.task.to_dict()["plan"]["terminal_reason"]
        == "RECOVERY_BUDGET_EXCEEDED"
    )


def test_non_candidate_navigation_id_is_rejected_without_pending_action() -> None:
    runtime = _runtime(_search_call(), _navigation_call("invented_poi"))

    answer = runtime.chat("帮我找附近服务区并导航", debug=False)

    assert "候选" in answer
    assert runtime.agent.pending_actions.get() is None
    assert runtime.agent.task.to_dict()["plan"]["status"] == PlanStatus.FAILED
    assert not any(
        item.name == "start_navigation" and item.confirmed
        for item in runtime.tools.execution_history()
    )


def test_rejecting_alternative_cancels_plan_without_second_navigation_write() -> None:
    runtime = _runtime(
        _search_call(),
        _navigation_call("rest_area_001"),
        unavailable_poi_ids=frozenset({"rest_area_001"}),
    )
    runtime.chat("帮我找附近服务区并导航", debug=False)
    original = runtime.agent.pending_actions.get()
    assert original is not None
    runtime.agent.confirm_pending(original.action_id)
    replacement = runtime.agent.pending_actions.get()
    assert replacement is not None

    rejected = runtime.agent.reject_pending(replacement.action_id)

    assert rejected.success is True
    assert runtime.agent.task.to_dict()["plan"]["status"] == PlanStatus.CANCELLED
    assert runtime.agent.pending_actions.get() is None
    assert (
        len(
            [
                item
                for item in runtime.tools.execution_history()
                if item.name == "start_navigation" and item.confirmed
            ]
        )
        == 1
    )


def test_retryable_location_search_failure_is_retried_once() -> None:
    runtime = _runtime(
        _search_call(),
        _navigation_call("rest_area_002"),
        transient_search_failures=1,
    )

    runtime.chat("帮我找附近服务区并导航", debug=False)

    searches = [
        item
        for item in runtime.tools.execution_history()
        if item.name == "search_nearby_rest_area"
    ]
    assert len(searches) == 2
    assert searches[0].error == "TRANSIENT_ERROR"
    assert searches[1].success is True
    assert runtime.agent.pending_actions.get().arguments["poi_id"] == "rest_area_002"


def test_plan_step_budget_exhaustion_stops_before_executing_next_tool() -> None:
    runtime = _runtime(_search_call(), ScriptedResponse(content="不应继续到这里。"))
    runtime.agent.plan_flow.config = AgentPlanConfig(
        max_steps=1, max_recoveries=0, max_candidates=3
    )

    answer = runtime.chat("帮我找附近服务区", debug=False)

    assert "步骤预算" in answer
    assert runtime.agent.task.to_dict()["plan"]["status"] == PlanStatus.FAILED
    assert (
        runtime.agent.task.to_dict()["plan"]["terminal_reason"]
        == "STEP_BUDGET_EXCEEDED"
    )
    assert not any(
        item.name == "search_nearby_rest_area"
        for item in runtime.tools.execution_history()
    )


def test_unavailable_only_candidate_stops_without_inventing_an_alternative() -> None:
    runtime = _runtime(_search_call())
    flow = runtime.agent.plan_flow
    plan = flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None
    plan.candidates = [{"poi_id": "rest_area_001", "name": "西湖服务区"}]
    plan.selected_poi_id = "rest_area_001"
    unavailable = ToolResult(False, "地点不可用", error="POI_UNAVAILABLE")

    result = flow._propose_alternative(runtime.agent, plan, unavailable)

    assert result is unavailable
    assert plan.status is PlanStatus.STOPPED_NO_RESULT
    assert plan.terminal_reason == "NO_ALTERNATIVE"
    assert runtime.agent.pending_actions.get() is None


def test_alternative_is_blocked_when_recovery_budget_is_exhausted() -> None:
    runtime = _runtime(_search_call())
    flow = runtime.agent.plan_flow
    flow.config = AgentPlanConfig(max_steps=9, max_recoveries=0, max_candidates=3)
    plan = flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None
    plan.candidates = [
        {"poi_id": "rest_area_001", "name": "首选"},
        {"poi_id": "rest_area_002", "name": "替代"},
    ]
    plan.selected_poi_id = "rest_area_001"
    unavailable = ToolResult(False, "地点不可用", error="POI_UNAVAILABLE")

    result = flow._propose_alternative(runtime.agent, plan, unavailable)

    assert result is unavailable
    assert plan.status is PlanStatus.FAILED
    assert plan.terminal_reason == "RECOVERY_BUDGET_EXCEEDED"
    assert runtime.agent.pending_actions.get() is None


def test_missing_pending_action_stops_candidate_staging(monkeypatch) -> None:
    runtime = _runtime(_search_call())
    flow = runtime.agent.plan_flow
    plan = flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None
    runtime.agent.confirmations.stage = lambda *_args, **_kwargs: None

    flow._stage_candidate(
        runtime.agent,
        plan,
        {"poi_id": "rest_area_001", "name": "西湖服务区"},
        reason="测试候选。",
    )

    assert plan.status is PlanStatus.FAILED
    assert plan.terminal_reason == "PENDING_ACTION_MISSING"
    assert plan.steps[-1].status.value == "FAILED"


def test_plan_flow_ignores_unrelated_intent_and_blocks_tools_after_terminal_state() -> (
    None
):
    runtime = _runtime(_search_call())
    flow = runtime.agent.plan_flow

    assert flow.begin(runtime.agent, "打开空调") is None
    assert flow.before_tool(runtime.agent, "search_nearby_rest_area", {}) is None

    plan = flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None
    assert flow.cancel(runtime.agent, "USER_CANCELLED") is True
    assert flow.before_tool(runtime.agent, "start_navigation", {"poi_id": "p1"})
    assert flow.cancel(runtime.agent, "ALREADY_TERMINAL") is False


def test_knowledge_step_records_success_and_failure_evidence() -> None:
    runtime = _runtime(_search_call())
    flow = runtime.agent.plan_flow
    plan = flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None

    assert flow.before_tool(runtime.agent, "search_vehicle_knowledge", {}) is None
    flow.observe_tool(
        runtime.agent,
        "search_vehicle_knowledge",
        {},
        ToolResult(True, "返回 1 条有来源的知识"),
    )
    assert plan.steps[-1].status.value == "SUCCEEDED"

    assert flow.before_tool(runtime.agent, "search_vehicle_knowledge", {}) is None
    flow.observe_tool(
        runtime.agent,
        "search_vehicle_knowledge",
        {},
        ToolResult(False, "暂时无法检索", error="RETRIEVAL_ERROR"),
    )
    assert plan.steps[-1].status.value == "FAILED"
    assert plan.steps[-1].error_code == "RETRIEVAL_ERROR"


def test_implicit_navigation_suggestion_only_stages_nearest_candidate() -> None:
    runtime = _runtime(_search_call())
    flow = runtime.agent.plan_flow
    plan = flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None
    plan.candidates = [
        {"poi_id": "far", "name": "远处", "distance_km": 12.0},
        {"poi_id": "near", "name": "近处", "distance_km": 3.0},
    ]
    runtime.agent.current_user_intent = "帮我找附近服务区并导航"

    flow.finish_turn(runtime.agent, "可以导航到最近的服务区，请确认。")

    pending = runtime.agent.pending_actions.get()
    assert pending is not None
    assert pending.arguments["poi_id"] == "near"
    assert not any(
        item.name == "start_navigation" and item.success
        for item in runtime.tools.execution_history()
    )


def test_read_retry_requires_remaining_plan_recovery_budget() -> None:
    runtime = _runtime(_search_call())
    flow = runtime.agent.plan_flow
    result = ToolResult(False, "暂时失败", error="TRANSIENT_ERROR")
    assert flow.claim_read_retry(runtime.agent, "search_nearby_rest_area", result)

    flow.config = AgentPlanConfig(max_steps=9, max_recoveries=0, max_candidates=3)
    plan = flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None
    assert not flow.claim_read_retry(runtime.agent, "search_nearby_rest_area", result)
    assert plan.status is PlanStatus.FAILED
    assert plan.terminal_reason == "RECOVERY_BUDGET_EXCEEDED"


def test_finish_confirmation_without_active_execute_step_is_ignored() -> None:
    runtime = _runtime(_search_call())
    plan = runtime.agent.plan_flow.begin(runtime.agent, "帮我找附近服务区")
    assert plan is not None

    result = runtime.agent.plan_flow.finish_confirmation(
        runtime.agent,
        {"tool_name": "start_navigation", "arguments": {"poi_id": "p1"}},
        ToolResult(True, "导航已启动"),
    )

    assert result is None
    assert plan.status is PlanStatus.RUNNING

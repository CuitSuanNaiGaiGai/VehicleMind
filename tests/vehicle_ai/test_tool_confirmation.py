from __future__ import annotations

import time
from threading import Event, Thread

from types import SimpleNamespace

import pytest

from modules.vehicle_ai.agent.action_state import PendingAction, PendingActionStore
from modules.vehicle_ai.context import ContextManager, NavigationState
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import build_default_tool_registry


@pytest.fixture
def runtime() -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient(responses=(ScriptedResponse(content="unused"),))
    )


def _pending_navigation(*, created_at: float | None = None) -> PendingAction:
    values = {}
    if created_at is not None:
        values["created_at"] = created_at
    return PendingAction(
        tool_name="start_navigation",
        arguments={"poi_id": "rest_area_001"},
        display_text="Navigate to West Lake Rest Area",
        expires_after_seconds=120.0,
        **values,
    )


def test_sensitive_tool_cannot_execute_without_confirmation(
    runtime: VehicleMindRuntime,
) -> None:
    result = runtime.tools.execute("start_navigation", {"poi_id": "rest_area_001"})

    assert result.success is False
    assert result.error == "CONFIRMATION_REQUIRED"
    assert (
        runtime.context_manager.get_context().vehicle.navigation_state
        == NavigationState.IDLE
    )


def test_non_sensitive_tool_executes_without_confirmation(
    runtime: VehicleMindRuntime,
) -> None:
    result = runtime.tools.execute("search_nearby_rest_area")

    assert result.success is True
    assert result.data["poi_id"] == "rest_area_001"


def test_confirmation_executes_only_original_action_once(
    runtime: VehicleMindRuntime,
) -> None:
    runtime.agent.pending_actions.set(_pending_navigation())
    pending = runtime.agent.pending_actions.get()
    assert pending is not None

    first = runtime.agent.confirm_pending(pending.action_id)
    second = runtime.agent.confirm_pending(pending.action_id)

    assert first.success is True
    assert second.success is False
    assert second.error == "INVALID_CONFIRMATION"
    vehicle = runtime.context_manager.get_context().vehicle
    assert vehicle.navigation_state == NavigationState.ACTIVE
    assert vehicle.navigation_destination_id == "rest_area_001"


def test_wrong_confirmation_id_does_not_consume_pending_action(
    runtime: VehicleMindRuntime,
) -> None:
    runtime.agent.pending_actions.set(_pending_navigation())
    pending = runtime.agent.pending_actions.get()
    assert pending is not None

    result = runtime.agent.confirm_pending("wrong-id")

    assert result.error == "INVALID_CONFIRMATION"
    assert runtime.agent.pending_actions.get() is not None


def test_rejection_requires_matching_action_id(runtime: VehicleMindRuntime) -> None:
    pending = _pending_navigation()
    runtime.agent.pending_actions.set(pending)

    wrong = runtime.agent.reject_pending("wrong-id")
    assert wrong.error == "INVALID_CONFIRMATION"
    assert runtime.agent.pending_actions.get() == pending

    rejected = runtime.agent.reject_pending(pending.action_id)
    assert rejected.success is True
    assert runtime.agent.pending_actions.get() is None
    assert (
        runtime.agent.confirm_pending(pending.action_id).error == "INVALID_CONFIRMATION"
    )
    assert (
        runtime.context_manager.get_context().vehicle.navigation_state
        == NavigationState.IDLE
    )


def test_pending_action_expires_at_exact_ttl_boundary() -> None:
    pending = _pending_navigation(created_at=100.0)

    assert pending.is_expired(now=220.0)


def test_rejection_cannot_erase_concurrently_staged_action() -> None:
    store = PendingActionStore()
    old = _pending_navigation()
    replacement = PendingAction(
        tool_name="set_driver_window", arguments={"open": True}, display_text="Open"
    )
    store.set(old)
    entered = Event()
    proceed = Event()
    setter_done = Event()
    original_get = store.get

    def paused_get() -> PendingAction | None:
        action = original_get()
        entered.set()
        assert proceed.wait(timeout=2)
        return action

    def set_replacement() -> None:
        store.set(replacement)
        setter_done.set()

    store.get = paused_get  # type: ignore[method-assign]
    rejecter = Thread(target=lambda: store.reject(old.action_id))
    setter = Thread(target=set_replacement)
    rejecter.start()
    assert entered.wait(timeout=2)
    setter.start()
    setter_done.wait(timeout=0.1)
    proceed.set()
    rejecter.join(timeout=2)
    setter.join(timeout=2)

    assert not rejecter.is_alive() and not setter.is_alive()
    assert store.get() == replacement


def test_pending_action_arguments_are_immutable_and_defensively_exported() -> None:
    pending = _pending_navigation()

    with pytest.raises(TypeError):
        pending.arguments["poi_id"] = "rest_area_002"  # type: ignore[index]

    exported = pending.to_dict()
    exported["arguments"]["poi_id"] = "rest_area_002"

    assert pending.arguments["poi_id"] == "rest_area_001"


def test_expired_pending_action_cannot_execute(runtime: VehicleMindRuntime) -> None:
    runtime.agent.pending_actions.set(
        _pending_navigation(created_at=time.time() - 121.0)
    )

    result = runtime.agent.confirm_pending("any-id")

    assert result.error == "INVALID_CONFIRMATION"
    assert runtime.agent.pending_actions.get() is None
    assert not runtime.tools.execution_history()


def test_confirmation_must_match_and_cannot_be_replayed() -> None:
    context = ContextManager()
    registry = build_default_tool_registry(context)
    issuer = registry.take_confirmation_issuer()
    grant = issuer.issue(
        "issued-action",
        "start_navigation",
        {"poi_id": "rest_area_001"},
    )

    mismatch = registry.execute(
        "start_navigation",
        {"poi_id": "rest_area_002"},
        confirmation=grant,
    )
    success = registry.execute(
        "start_navigation",
        {"poi_id": "rest_area_001"},
        confirmation=grant,
    )
    replay = registry.execute(
        "start_navigation",
        {"poi_id": "rest_area_001"},
        confirmation=grant,
    )

    assert mismatch.error == "CONFIRMATION_MISMATCH"
    assert success.success is True
    assert replay.error == "CONFIRMATION_REPLAY"


def test_structurally_similar_confirmation_cannot_authorize_execution(
    runtime: VehicleMindRuntime,
) -> None:
    forged = SimpleNamespace(
        action_id="forged",
        tool_name="set_driver_window",
        arguments={"open": True},
    )

    result = runtime.tools.execute(
        "set_driver_window",
        {"open": True},
        confirmation=forged,
    )

    assert result.success is False
    assert result.error == "INVALID_CONFIRMATION"
    assert runtime.context_manager.get_context().vehicle.driver_window_open is False
    assert runtime.tools.execution_history()[-1].confirmed is False


def test_runtime_registry_does_not_expose_another_issuer(
    runtime: VehicleMindRuntime,
) -> None:
    with pytest.raises(RuntimeError, match="already been claimed"):
        runtime.tools.take_confirmation_issuer()
    with pytest.raises(PermissionError, match="invalid confirmation issuer"):
        runtime.tools._issue_confirmation(
            object(),
            "attacker-minted",
            "set_driver_window",
            {"open": True},
        )


def test_tool_history_records_blocked_and_successful_attempts(
    runtime: VehicleMindRuntime,
) -> None:
    runtime.tools.execute("start_navigation", {"poi_id": "rest_area_001"})
    runtime.tools.execute("search_nearby_rest_area")

    history = runtime.tools.execution_history()

    assert [(item.name, item.success, item.error) for item in history] == [
        ("start_navigation", False, "CONFIRMATION_REQUIRED"),
        ("search_nearby_rest_area", True, None),
    ]


def test_agent_stages_sensitive_tool_call_instead_of_executing() -> None:
    llm = ScriptedLLMClient(
        responses=(
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="call-window",
                        name="set_driver_window",
                        arguments={"open": True},
                        arguments_json='{"open": true}',
                    ),
                ),
            ),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)

    answer = runtime.chat("Open my window", debug=False)

    pending = runtime.agent.pending_actions.get()
    assert answer == "车机操作已准备好，尚未执行，待确认后才会执行。"
    assert pending is not None
    assert pending.tool_name == "set_driver_window"
    assert pending.arguments == {"open": True}
    assert runtime.context_manager.get_context().vehicle.driver_window_open is False


def test_confirmation_required_stops_tool_loop_and_does_not_claim_failure() -> None:
    llm = ScriptedLLMClient(
        responses=(
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="call-window",
                        name="set_driver_window",
                        arguments={"open": True},
                        arguments_json='{"open": true}',
                    ),
                ),
            ),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)

    answer = runtime.chat("打开驾驶员车窗", debug=False)

    assert "待确认" in answer
    assert "尚未执行" in answer
    assert "失败" not in answer
    assert len(llm.requests) == 1
    assert runtime.agent.pending_actions.get() is not None
    assert runtime.context_manager.get_context().vehicle.driver_window_open is False
    assert runtime.tools.execution_history()[-1].error == "CONFIRMATION_REQUIRED"


def test_matching_pending_survives_other_blocked_call_in_same_batch() -> None:
    llm = ScriptedLLMClient(
        responses=(
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall("window", "set_driver_window", {"open": True},
                                '{"open": true}'),
                    LLMToolCall("nav", "start_navigation",
                                {"poi_id": "rest_area_001"},
                                '{"poi_id": "rest_area_001"}'),
                ),
            ),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)

    answer = runtime.chat("打开车窗，并导航到服务区", debug=False)

    assert "待确认" in answer
    assert len(llm.requests) == 1
    pending = runtime.agent.pending_actions.get()
    assert pending is not None and pending.tool_name == "set_driver_window"
    assert len(runtime.tools.execution_history()) == 1


def test_sensitive_pending_cannot_be_replaced_by_later_search_in_batch() -> None:
    llm = ScriptedLLMClient(
        responses=(
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall("window", "set_driver_window", {"open": True},
                                '{"open": true}'),
                    LLMToolCall("search", "search_nearby_rest_area", {}, "{}"),
                ),
            ),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)

    answer = runtime.chat("打开车窗并找服务区", debug=False)

    assert "待确认" in answer
    pending = runtime.agent.pending_actions.get()
    assert pending is not None and pending.tool_name == "set_driver_window"
    assert [item.name for item in runtime.tools.execution_history()] == [
        "set_driver_window"
    ]


def test_explicit_refusal_clears_pending_without_calling_llm() -> None:
    llm = ScriptedLLMClient((ScriptedResponse(content="unused"),))
    runtime = VehicleMindRuntime(llm=llm)
    pending = _pending_navigation()
    runtime.agent.pending_actions.set(pending)

    answer = runtime.chat("不要了", debug=False)

    assert answer
    assert llm.remaining == 1
    assert runtime.agent.pending_actions.get() is None
    assert (
        runtime.agent.confirm_pending(pending.action_id).error == "INVALID_CONFIRMATION"
    )
    assert not runtime.tools.execution_history()


@pytest.mark.parametrize("user_text", ["取消", "换成西湖服务区"])
def test_chat_does_not_claim_cancellation_of_replaced_action(user_text: str) -> None:
    llm = ScriptedLLMClient((ScriptedResponse(content="unused"),))
    runtime = VehicleMindRuntime(llm=llm)
    old = _pending_navigation()
    replacement = PendingAction(
        tool_name="set_driver_window", arguments={"open": True}, display_text="Open"
    )
    runtime.agent.pending_actions.set(old)
    original_reject = runtime.agent.confirmations.reject

    def replace_before_reject(action_id: str):
        runtime.agent.pending_actions.set(replacement)
        return original_reject(action_id)

    runtime.agent.confirmations.reject = replace_before_reject  # type: ignore[method-assign]

    answer = runtime.chat(user_text, debug=False)

    assert answer == "待确认操作已变更，请重新确认当前操作。"
    assert runtime.agent.pending_actions.get() == replacement
    assert llm.remaining == 1


def test_target_change_does_not_ground_new_call_to_old_destination() -> None:
    llm = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="new-target",
                        name="start_navigation",
                        arguments={"poi_id": "rest_area_002"},
                        arguments_json='{"poi_id": "rest_area_002"}',
                    ),
                ),
            ),
            ScriptedResponse(content="Please confirm the new destination."),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)
    old = _pending_navigation()
    runtime.agent.pending_actions.set(old)

    answer = runtime.chat("换成东湖服务区", debug=False)

    assert runtime.agent.pending_actions.get() is None
    assert answer == "未找到与新目标匹配的地点，未创建待确认导航。"
    assert runtime.agent.confirm_pending(old.action_id).error == "INVALID_CONFIRMATION"
    assert (
        runtime.context_manager.get_context().vehicle.navigation_state
        == NavigationState.IDLE
    )


def test_target_change_replaces_action_only_after_search_result() -> None:
    llm = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="search-new",
                        name="search_nearby_rest_area",
                        arguments={},
                        arguments_json="{}",
                    ),
                ),
            ),
            ScriptedResponse(content="Please confirm the new destination."),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)
    old = PendingAction(
        tool_name="start_navigation",
        arguments={"poi_id": "rest_area_002"},
        display_text="Navigate to Riverside Service Area",
    )
    runtime.agent.pending_actions.set(old)

    runtime.chat("换成西湖服务区", debug=False)

    current = runtime.agent.pending_actions.get()
    assert current is not None
    assert current.action_id != old.action_id
    assert current.arguments["poi_id"] == "rest_area_001"
    assert runtime.agent.confirm_pending(old.action_id).error == "INVALID_CONFIRMATION"


def test_target_change_does_not_stage_search_result_for_different_destination() -> None:
    llm = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="search-wrong",
                        name="search_nearby_rest_area",
                        arguments={},
                        arguments_json="{}",
                    ),
                ),
            ),
            ScriptedResponse(content="Requested destination not found."),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)
    old = _pending_navigation()
    runtime.agent.pending_actions.set(old)

    runtime.chat("换成东湖服务区", debug=False)

    assert runtime.agent.pending_actions.get() is None
    assert runtime.agent.confirm_pending(old.action_id).error == "INVALID_CONFIRMATION"


def test_navigation_call_cannot_replace_unrelated_pending_action() -> None:
    llm = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="invented-nav",
                        name="start_navigation",
                        arguments={"poi_id": "rest_area_002"},
                        arguments_json='{"poi_id": "rest_area_002"}',
                    ),
                ),
            ),
            ScriptedResponse(content="Navigation requires a search."),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)
    window = PendingAction(
        tool_name="set_driver_window",
        arguments={"open": True},
        display_text="Open driver window",
    )
    runtime.agent.pending_actions.set(window)

    runtime.chat("导航到附近", debug=False)

    assert runtime.agent.pending_actions.get() == window


def test_hostile_tool_call_after_rejection_cannot_restore_old_navigation() -> None:
    llm = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        id="restore-old",
                        name="start_navigation",
                        arguments={"poi_id": "rest_area_001"},
                        arguments_json='{"poi_id": "rest_area_001"}',
                    ),
                ),
            ),
            ScriptedResponse(content="Navigation was not started."),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)
    old = _pending_navigation()
    runtime.agent.pending_actions.set(old)

    runtime.chat("取消", debug=False)
    runtime.chat("现在车况怎么样", debug=False)

    assert runtime.agent.pending_actions.get() is None
    assert runtime.agent.confirm_pending(old.action_id).error == "INVALID_CONFIRMATION"
    sensitive_attempts = [
        record
        for record in runtime.tools.execution_history()
        if record.requires_confirmation
    ]
    assert len(sensitive_attempts) == 1
    assert sensitive_attempts[0].error == "CONFIRMATION_REQUIRED"
    assert sensitive_attempts[0].success is False

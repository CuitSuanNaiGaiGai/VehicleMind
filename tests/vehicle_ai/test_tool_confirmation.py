from __future__ import annotations

import time

import pytest

from modules.vehicle_ai.agent.action_state import PendingAction
from modules.vehicle_ai.context import NavigationState
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime


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


def test_confirmation_must_match_and_cannot_be_replayed(
    runtime: VehicleMindRuntime,
) -> None:
    runtime.agent.pending_actions.set(_pending_navigation())
    pending = runtime.agent.pending_actions.get()
    assert pending is not None
    confirmation = runtime.agent.pending_actions.consume(pending.action_id)
    assert confirmation is not None

    mismatch = runtime.tools.execute(
        "start_navigation",
        {"poi_id": "rest_area_002"},
        confirmation=confirmation,
    )
    success = runtime.tools.execute(
        "start_navigation",
        {"poi_id": "rest_area_001"},
        confirmation=confirmation,
    )
    replay = runtime.tools.execute(
        "start_navigation",
        {"poi_id": "rest_area_001"},
        confirmation=confirmation,
    )

    assert mismatch.error == "CONFIRMATION_MISMATCH"
    assert success.success is True
    assert replay.error == "CONFIRMATION_REPLAY"


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
            ScriptedResponse(content="Please confirm opening the window."),
        )
    )
    runtime = VehicleMindRuntime(llm=llm)

    answer = runtime.chat("Open my window", debug=False)

    pending = runtime.agent.pending_actions.get()
    assert answer == "Please confirm opening the window."
    assert pending is not None
    assert pending.tool_name == "set_driver_window"
    assert pending.arguments == {"open": True}
    assert runtime.context_manager.get_context().vehicle.driver_window_open is False

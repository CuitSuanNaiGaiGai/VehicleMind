from __future__ import annotations

import time

from types import SimpleNamespace

import pytest

from modules.vehicle_ai.agent.action_state import PendingAction
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
    assert runtime.agent.confirm_pending(pending.action_id).error == "INVALID_CONFIRMATION"
    assert runtime.context_manager.get_context().vehicle.navigation_state == NavigationState.IDLE


def test_pending_action_expires_at_exact_ttl_boundary() -> None:
    pending = _pending_navigation(created_at=100.0)

    assert pending.is_expired(now=220.0)


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

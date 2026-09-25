from __future__ import annotations

from modules.vehicle_ai.context.enums import RiskLevel
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import build_default_tool_registry


def runtime(*responses: ScriptedResponse) -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient(
            responses=responses or (ScriptedResponse(content="unused"),)
        )
    )


def test_registry_rechecks_tool_flags_and_denies_before_handler() -> None:
    app = runtime()
    tool = app.tools.get("play_music")
    tool.read_only = True

    result = app.tools.execute(
        "play_music", {"query": "轻音乐"}, user_intent="放点音乐"
    )

    assert result.error == "POLICY_DENIED"
    assert app.context_manager.get_context().vehicle.media_playing is False
    assert result.to_dict()["policy"]["decision"] == "DENY"
    assert "read-only" in result.to_dict()["policy"]["reason"].lower()
    assert app.tools.execution_history()[-1].policy == result.to_dict()["policy"]


def test_known_high_risk_music_executes_with_rest_warning() -> None:
    app = runtime()
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    result = app.tools.execute(
        "play_music", {"query": "轻音乐"}, user_intent="放点音乐"
    )

    assert result.success
    assert app.context_manager.get_context().vehicle.media_playing is True
    assert result.to_dict()["policy"]["decision"] == "ALLOW"
    assert result.to_dict()["policy"]["risk"] == "REVERSIBLE_WRITE"
    assert any("停车休息" in text for text in result.to_dict()["policy"]["warnings"])


def test_missing_driver_state_does_not_assert_fatigue() -> None:
    result = runtime().tools.execute(
        "play_music", {"query": "轻音乐"}, user_intent="放点音乐"
    )

    assert result.success
    warnings = result.to_dict()["policy"]["warnings"]
    assert any("未知" in text or "不可用" in text for text in warnings)
    assert all("疲劳" not in text for text in warnings)


def test_confirmation_is_one_use_and_policy_is_recorded() -> None:
    registry = build_default_tool_registry(ContextManager())
    issuer = registry.take_confirmation_issuer()
    grant = issuer.issue("one-use", "set_driver_window", {"open": True})

    blocked = registry.execute("set_driver_window", {"open": True})
    allowed = registry.execute("set_driver_window", {"open": True}, confirmation=grant)
    replay = registry.execute("set_driver_window", {"open": True}, confirmation=grant)

    assert blocked.error == "CONFIRMATION_REQUIRED"
    assert blocked.to_dict()["policy"]["decision"] == "REQUIRE_CONFIRMATION"
    assert (
        allowed.success
        and allowed.to_dict()["policy"]["decision"] == "REQUIRE_CONFIRMATION"
    )
    assert replay.error == "CONFIRMATION_REPLAY"
    assert registry.execution_history()[-1].confirmed is False


def test_confirmation_rechecks_current_state_and_preserves_original_intent() -> None:
    app = runtime()
    app.context_manager.update_driver(risk=RiskLevel.LOW)
    app.agent.confirmations.stage(
        "set_driver_window", {"open": True}, user_intent="打开驾驶员车窗"
    )
    pending = app.agent.pending_actions.get()
    assert pending is not None
    assert pending.metadata["user_intent"] == "打开驾驶员车窗"
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    result = app.agent.confirm_pending(pending.action_id)

    assert result.success
    assert any("停车休息" in text for text in result.to_dict()["policy"]["warnings"])
    assert app.tools.execution_history()[-1].user_intent == "打开驾驶员车窗"


def test_confirmation_policy_denial_does_not_run_handler_or_reuse_grant() -> None:
    app = runtime()
    app.agent.confirmations.stage("set_driver_window", {"open": True})
    pending = app.agent.pending_actions.get()
    assert pending is not None
    app.tools.get("set_driver_window").read_only = True

    denied = app.agent.confirm_pending(pending.action_id)
    repeated = app.agent.confirm_pending(pending.action_id)

    assert denied.error == "POLICY_DENIED"
    assert repeated.error == "INVALID_CONFIRMATION"
    assert app.context_manager.get_context().vehicle.driver_window_open is False
    assert not app.tools._issued_confirmations
    policy = app.tools.execution_history()[-1].policy
    assert policy is not None and policy["decision"] == "DENY"


def test_agent_trace_contains_policy_and_current_user_intent() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(
                LLMToolCall(
                    id="music",
                    name="play_music",
                    arguments={"query": "轻音乐"},
                    arguments_json='{"query": "轻音乐"}',
                ),
            ),
        ),
        ScriptedResponse(content="已播放音乐。"),
    )
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    app.chat("放点音乐", debug=False)

    result_events = [
        event for event in app.agent.trace if event["kind"] == "tool_result"
    ]
    assert result_events
    assert result_events[-1]["result"]["policy"]["decision"] == "ALLOW"
    assert any(
        "停车休息" in text for text in result_events[-1]["result"]["policy"]["warnings"]
    )
    assert app.tools.execution_history()[-1].user_intent == "放点音乐"


def test_high_risk_music_reply_includes_fixed_rest_advice_even_if_model_omits_it() -> (
    None
):
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(
                LLMToolCall(
                    id="music",
                    name="play_music",
                    arguments={"query": "轻音乐"},
                    arguments_json='{"query": "轻音乐"}',
                ),
            ),
        ),
        ScriptedResponse(content="已播放音乐。"),
    )
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    answer = app.chat("放点音乐", debug=False)

    assert "停车休息" in answer
    assert "音乐不能消除疲劳" in answer
    assert "音乐不能替代休息" in answer


def test_high_risk_music_reply_does_not_repeat_unsafe_model_claim() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(
                LLMToolCall(
                    id="music",
                    name="play_music",
                    arguments={"query": "轻音乐"},
                    arguments_json='{"query": "轻音乐"}',
                ),
            ),
        ),
        ScriptedResponse(content="已播放音乐，音乐可以消除疲劳。"),
    )
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    answer = app.chat("放点音乐", debug=False)

    assert "音乐可以消除疲劳" not in answer
    assert "音乐不能消除疲劳" in answer
    assert "停车休息" in answer


def test_provider_tool_schema_is_unchanged_by_policy_metadata() -> None:
    app = runtime()
    schema = app.tools.get("play_music").llm_schema()

    assert set(schema) == {"type", "function"}
    assert set(schema["function"]) == {"name", "description", "parameters"}

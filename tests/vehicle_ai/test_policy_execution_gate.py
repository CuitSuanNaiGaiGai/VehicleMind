from __future__ import annotations

from threading import Event, Thread, current_thread

import pytest

from modules.observation import ObservationMetadata
from modules.vehicle_ai.context.enums import RiskLevel
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.context.quality import QualityStatus
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.vehicle_ai.tools import build_default_tool_registry
from modules.vehicle_ai.tools.base import ToolResult


def runtime(*responses: ScriptedResponse) -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient(
            responses=responses or (ScriptedResponse(content="unused"),)
        )
    )


def call(name: str, arguments: dict | None = None) -> LLMToolCall:
    arguments = arguments or {}
    encoded = (
        '{"query": "轻音乐"}'
        if name == "play_music"
        else ('{"open": true}' if name == "set_driver_window" else "{}")
    )
    return LLMToolCall(name, name, arguments, encoded)


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


def test_high_risk_play_then_pause_reports_paused_terminal_state() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(call("play_music", {"query": "轻音乐"}), call("pause_music")),
        ),
        ScriptedResponse(content="音乐正在播放。"),
    )
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    answer = app.chat("播放后暂停音乐", debug=False)

    assert app.agent.task.status == "COMPLETED"
    assert app.context_manager.get_context().vehicle.media_playing is False
    assert "已暂停" in answer
    assert "已开始播放音乐" not in answer
    assert "音乐不能消除疲劳" in answer


def test_high_risk_play_then_second_tool_failure_reports_incomplete_turn() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(call("play_music", {"query": "轻音乐"}), call("pause_music")),
        ),
        ScriptedResponse(content="全部成功。"),
    )
    app.context_manager.update_driver(risk=RiskLevel.HIGH)
    app.tools.get("pause_music").handler = lambda: ToolResult(
        False, "Pause failed.", error="PAUSE_FAILED"
    )

    answer = app.chat("播放后暂停音乐", debug=False)

    assert app.agent.task.status == "FAILED"
    assert app.agent.task.reason == "PAUSE_FAILED"
    assert "播放成功" in answer
    assert "未完成" in answer
    assert "PAUSE_FAILED" in answer
    assert "已暂停" not in answer
    assert "音乐不能消除疲劳" in answer


def test_high_risk_play_then_model_failure_reports_incomplete_turn() -> None:
    app = runtime(
        ScriptedResponse(
            content=None, tool_calls=(call("play_music", {"query": "轻音乐"}),)
        )
    )
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    answer = app.chat("放音乐", debug=False)

    assert app.agent.task.status == "FAILED"
    assert app.agent.task.reason == "MODEL_ERROR"
    assert "播放成功" in answer
    assert "未完成" in answer
    assert "MODEL_ERROR" in answer
    assert "音乐不能消除疲劳" in answer


def test_high_risk_play_then_pending_action_reports_unexecuted_action() -> None:
    app = runtime(
        ScriptedResponse(
            content=None,
            tool_calls=(
                call("play_music", {"query": "轻音乐"}),
                call("set_driver_window", {"open": True}),
            ),
        ),
    )
    app.context_manager.update_driver(risk=RiskLevel.HIGH)

    answer = app.chat("放音乐并打开车窗", debug=False)

    assert app.agent.task.status == "AWAITING_CONFIRMATION"
    assert app.context_manager.get_context().vehicle.driver_window_open is False
    assert "播放成功" in answer
    assert "尚未执行" in answer
    assert "待确认" in answer
    assert "音乐不能消除疲劳" in answer


def test_unknown_tool_retires_live_grant_before_replay() -> None:
    registry = build_default_tool_registry(ContextManager())
    issuer = registry.take_confirmation_issuer()
    grant = issuer.issue("unknown-attempt", "set_driver_window", {"open": True})

    denied = registry.execute("no_such_tool", {}, confirmation=grant)
    replay = registry.execute("set_driver_window", {"open": True}, confirmation=grant)

    assert denied.error == "UNKNOWN_TOOL"
    assert replay.error == "CONFIRMATION_REPLAY"
    assert not registry._issued_confirmations


def test_policy_snapshot_uses_one_quality_time_and_excludes_interleaving_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_reads: dict[str, int] = {}

    def quality_clock() -> float:
        name = current_thread().name
        clock_reads[name] = clock_reads.get(name, 0) + 1
        return 10.0

    manager = ContextManager(quality_clock=quality_clock)
    manager.update_driver(risk=RiskLevel.HIGH)
    manager.update_vehicle(speed_kmh=30.0)
    snapshot_with_quality = manager.snapshot_with_quality
    entered_quality = Event()
    release_quality = Event()
    writer_started = Event()
    writer_finished = Event()
    original = manager._quality.field_status
    captured: list[tuple] = []

    def held_status(*args, **kwargs):
        entered_quality.set()
        assert release_quality.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(manager._quality, "field_status", held_status)

    def read_snapshot() -> None:
        captured.append(
            snapshot_with_quality((("driver", "risk"), ("vehicle", "speed_kmh")))
        )

    def invalidate_driver() -> None:
        writer_started.set()
        manager.mark_invalid_observation(
            "driver",
            ObservationMetadata(
                timestamp_ms=1000,
                sequence=0,
                source="test",
                confidence=None,
                valid=False,
                processing_ms=0.0,
            ),
        )
        manager.update_driver(risk=RiskLevel.LOW)
        writer_finished.set()

    reader = Thread(target=read_snapshot, name="snapshot-reader")
    writer = Thread(target=invalidate_driver)
    reader.start()
    try:
        assert entered_quality.wait(2)
        writer.start()
        assert writer_started.wait(2)
        assert not writer_finished.wait(0.05)
    finally:
        release_quality.set()
        reader.join(2)
        if writer.ident is not None:
            writer.join(2)

    assert writer_finished.is_set()
    context, qualities = captured[0]
    assert context.driver.risk is RiskLevel.HIGH
    assert qualities[("driver", "risk")] is QualityStatus.KNOWN
    assert qualities[("vehicle", "speed_kmh")] is QualityStatus.KNOWN
    assert clock_reads["snapshot-reader"] == 1


def test_registry_policy_provider_uses_atomic_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ContextManager()
    manager.update_driver(risk=RiskLevel.HIGH)
    registry = build_default_tool_registry(manager)
    monkeypatch.setattr(
        manager,
        "get_context",
        lambda: (_ for _ in ()).throw(AssertionError("split context read")),
    )
    monkeypatch.setattr(
        manager,
        "field_quality",
        lambda *_args: (_ for _ in ()).throw(AssertionError("split quality read")),
    )

    result = registry.execute("set_volume", {"volume": 20})

    assert result.success
    assert result.policy is not None
    assert any("停车休息" in warning for warning in result.policy["warnings"])


def test_provider_tool_schema_is_unchanged_by_policy_metadata() -> None:
    app = runtime()
    schema = app.tools.get("play_music").llm_schema()

    assert set(schema) == {"type", "function"}
    assert set(schema["function"]) == {"name", "description", "parameters"}

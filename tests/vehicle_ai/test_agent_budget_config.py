from pathlib import Path

import pytest

from modules.vehicle_ai.agent.budget import AgentBudgetConfig


def test_repository_agent_limits_load_from_yaml() -> None:
    config = AgentBudgetConfig.load()
    assert config.turn_timeout_seconds == 90
    assert config.max_tool_calls == 10
    assert config.max_tool_rounds == 5
    assert config.max_task_trace_events == 200


def test_invalid_agent_limits_are_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("turn_timeout_seconds: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        AgentBudgetConfig.load(config_path)
    config_path.write_text(
        "turn_timeout_seconds: 3\nmax_tool_calls: 2\nmax_tool_rounds: 1\nmax_task_trace_events: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive"):
        AgentBudgetConfig.load(config_path)


@pytest.mark.parametrize("provider", ["qwen", "glm"])
def test_provider_forwards_remaining_turn_timeout(provider, monkeypatch) -> None:
    from types import SimpleNamespace

    if provider == "qwen":
        from modules.vehicle_ai.llm.qwen_client import QwenClient as Client

        client = Client(api_key="test", timeout_seconds=30)
    else:
        from modules.vehicle_ai.llm.glm_client import GLMClient as Client

        client = Client(api_key="test", timeout_seconds=30)

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="test-model",
            usage=None,
        )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    client.chat_with_timeout([], timeout_seconds=7.25)
    assert captured["timeout"] == 7.25

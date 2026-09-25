from __future__ import annotations

import pytest

from modules.config.agent_plan import AgentPlanConfig


def test_agent_plan_config_loads_bounded_defaults() -> None:
    config = AgentPlanConfig.load()

    assert config.max_steps == 9
    assert config.max_recoveries == 1
    assert config.max_candidates == 3


def test_agent_plan_config_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        AgentPlanConfig.load(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "raw, message",
    [
        ("", "invalid fields"),
        (
            "schema_version: 1\nmax_steps: 9\nmax_recoveries: 1\nmax_candidates: 3\nextra: true\n",
            "invalid fields",
        ),
        (
            "schema_version: 1\nmax_steps: 0\nmax_recoveries: 1\nmax_candidates: 3\n",
            "positive integer",
        ),
        (
            "schema_version: 1\nmax_steps: 9\nmax_recoveries: 1.0\nmax_candidates: 3\n",
            "positive integer",
        ),
    ],
)
def test_agent_plan_config_rejects_invalid_values(tmp_path, raw, message) -> None:
    path = tmp_path / "agent_plan.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        AgentPlanConfig.load(path)

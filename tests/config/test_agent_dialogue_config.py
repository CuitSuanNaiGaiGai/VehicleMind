from pathlib import Path

import pytest

from modules.config.agent_dialogue import AgentDialogueConfig


def test_agent_dialogue_config_defaults_disabled_and_loads_independent_files(
    tmp_path: Path,
) -> None:
    default = AgentDialogueConfig.load()
    assert default.enabled is False
    assert default.candidate_ttl_seconds == 120
    assert default.max_revisions == 3
    assert default.interpreter_timeout_seconds == 15

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        "schema_version: 1\nenabled: true\ncandidate_ttl_seconds: 45\n"
        "max_revisions: 2\ninterpreter_timeout_seconds: 8\n",
        encoding="utf-8",
    )
    second.write_text(
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 30\n"
        "max_revisions: 1\ninterpreter_timeout_seconds: 5\n",
        encoding="utf-8",
    )

    first_config = AgentDialogueConfig.load(first)
    second_config = AgentDialogueConfig.load(str(second))
    assert first_config.enabled is True
    assert first_config.candidate_ttl_seconds == 45
    assert second_config.enabled is False
    assert second_config.candidate_ttl_seconds == 30
    assert AgentDialogueConfig.load() == default


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\nextra: true\n",
    ],
)
def test_agent_dialogue_config_rejects_missing_or_unknown_keys(
    tmp_path: Path, raw: str
) -> None:
    path = tmp_path / "agent_dialogue.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError):
        AgentDialogueConfig.load(path)


@pytest.mark.parametrize(
    "raw",
    [
        "schema_version: true\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 2\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: 1\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: true\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: .nan\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: .inf\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 0\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120.1\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: true\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 0\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 4\ninterpreter_timeout_seconds: 15\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: false\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: .inf\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 0\n",
        "schema_version: 1\nenabled: false\ncandidate_ttl_seconds: 120\n"
        "max_revisions: 3\ninterpreter_timeout_seconds: 15.1\n",
    ],
)
def test_agent_dialogue_config_rejects_invalid_types_and_bounds(
    tmp_path: Path, raw: str
) -> None:
    path = tmp_path / "agent_dialogue.yaml"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError):
        AgentDialogueConfig.load(path)

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modules.vehicle_ai.agent.policy import (
    ActionRisk,
    AgentPolicy,
    PolicyContext,
    PolicyDecision,
)
from modules.vehicle_ai.context.enums import RiskLevel
from modules.vehicle_ai.context.quality import QualityStatus
from modules.vehicle_ai.tools.base import ToolDefinition, ToolResult


CONFIG = Path(__file__).parents[2] / "modules/config/agent_policy.yaml"


def tool(
    name: str, *, confirmed: bool = False, read_only: bool = False
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        handler=lambda: ToolResult(True, "ok"),
        requires_confirmation=confirmed,
        read_only=read_only,
    )


def context(
    risk: RiskLevel = RiskLevel.LOW,
    quality: QualityStatus = QualityStatus.KNOWN,
    *,
    moving: bool = True,
    intent: str = "放点音乐",
) -> PolicyContext:
    return PolicyContext(risk, quality, moving, intent)


@pytest.mark.parametrize(
    ("definition", "risk", "decision"),
    [
        (
            tool("get_media_status", read_only=True),
            ActionRisk.READ_ONLY,
            PolicyDecision.ALLOW,
        ),
        (tool("play_music"), ActionRisk.REVERSIBLE_WRITE, PolicyDecision.ALLOW),
        (
            tool("start_navigation", confirmed=True),
            ActionRisk.CONFIRMATION_REQUIRED,
            PolicyDecision.REQUIRE_CONFIRMATION,
        ),
    ],
)
def test_configured_tool_risk_and_decision(
    definition: ToolDefinition, risk: ActionRisk, decision: PolicyDecision
) -> None:
    result = AgentPolicy.from_yaml(CONFIG).evaluate(definition, context())
    assert result.risk is risk
    assert result.decision is decision
    assert result.reason


@pytest.mark.parametrize(
    "quality",
    [
        QualityStatus.MISSING,
        QualityStatus.INVALID,
        QualityStatus.STALE,
        QualityStatus.UNKNOWN,
    ],
)
def test_untrusted_driver_state_does_not_create_fatigue_claim(
    quality: QualityStatus,
) -> None:
    result = AgentPolicy.from_yaml(CONFIG).evaluate(
        tool("play_music"), context(RiskLevel.HIGH, quality)
    )
    assert result.decision is PolicyDecision.ALLOW
    assert not any("疲劳" in warning for warning in result.warnings)
    assert any("未知" in warning or "不可用" in warning for warning in result.warnings)


def test_known_high_risk_music_is_allowed_with_fatigue_warning() -> None:
    result = AgentPolicy.from_yaml(CONFIG).evaluate(
        tool("play_music"), context(RiskLevel.HIGH)
    )
    assert result.decision is PolicyDecision.ALLOW
    assert any(
        "疲劳" in warning and "停车休息" in warning for warning in result.warnings
    )
    assert not any("消除疲劳" in warning for warning in result.warnings)


def test_navigation_still_requires_confirmation_under_high_risk() -> None:
    result = AgentPolicy.from_yaml(CONFIG).evaluate(
        tool("start_navigation", confirmed=True), context(RiskLevel.HIGH)
    )
    assert result.decision is PolicyDecision.REQUIRE_CONFIRMATION


def test_unknown_tool_is_denied() -> None:
    result = AgentPolicy.from_yaml(CONFIG).evaluate(tool("unlisted_action"), context())
    assert result.decision is PolicyDecision.DENY


def test_tool_confirmation_flag_cannot_be_downgraded_by_config() -> None:
    result = AgentPolicy.from_yaml(CONFIG).evaluate(
        tool("play_music", confirmed=True), context()
    )
    assert result.decision is PolicyDecision.DENY


def test_writable_tool_reclassified_as_read_only_is_denied(tmp_path: Path) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["tool_risks"]["play_music"] = "READ_ONLY"
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = AgentPolicy.from_yaml(path).evaluate(tool("play_music"), context())
    assert result.decision is PolicyDecision.DENY
    assert "Read-only action is allowed" not in result.reason


def test_read_only_tool_reclassified_as_write_is_denied(tmp_path: Path) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["tool_risks"]["get_media_status"] = "REVERSIBLE_WRITE"
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = AgentPolicy.from_yaml(path).evaluate(
        tool("get_media_status", read_only=True), context()
    )
    assert result.decision is PolicyDecision.DENY


def test_confirmation_required_tool_downgraded_in_yaml_is_denied(
    tmp_path: Path,
) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["tool_risks"]["start_navigation"] = "REVERSIBLE_WRITE"
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = AgentPolicy.from_yaml(path).evaluate(
        tool("start_navigation", confirmed=True), context()
    )
    assert result.decision is PolicyDecision.DENY


def test_non_confirmation_tool_upgraded_in_yaml_is_denied(tmp_path: Path) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["tool_risks"]["play_music"] = "CONFIRMATION_REQUIRED"
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = AgentPolicy.from_yaml(path).evaluate(tool("play_music"), context())
    assert result.decision is PolicyDecision.DENY


@pytest.mark.parametrize("warning_key", ["fatigue_warning", "unknown_driver_warning"])
def test_warning_prose_cannot_be_injected_from_yaml(
    tmp_path: Path, warning_key: str
) -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["risk_policy"][warning_key] = "音乐可以消除疲劳"
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        AgentPolicy.from_yaml(path)


def test_default_warning_policy_uses_versioned_codes() -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["risk_policy"] == {
        "high_driver_risk": "HIGH",
        "fatigue_warning_code": "REST_REQUIRED",
        "unknown_driver_warning_code": "STATE_UNAVAILABLE",
    }
    AgentPolicy.from_yaml(CONFIG)


@pytest.mark.parametrize(
    "body",
    [
        "schema_version: 2\ntool_risks: {}\nrisk_policy: {}\n",
        "schema_version: 1\ntool_risks: {}\nrisk_policy: {}\nextra: true\n",
        "schema_version: 1\ntool_risks: {play_music: DANGEROUS}\nrisk_policy: {}\n",
        "schema_version: 1\ntool_risks: {play_music: REVERSIBLE_WRITE}\nrisk_policy: {fatigue_warning: 4}\n",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, body: str) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        AgentPolicy.from_yaml(path)

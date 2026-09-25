"""Deterministic tool policy based on configured action risk and trusted state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from modules.vehicle_ai.context.enums import RiskLevel
from modules.vehicle_ai.context.quality import QualityStatus
from modules.vehicle_ai.tools.base import ToolDefinition


class ActionRisk(StrEnum):
    READ_ONLY = "READ_ONLY"
    REVERSIBLE_WRITE = "REVERSIBLE_WRITE"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"


class PolicyDecision(StrEnum):
    ALLOW = "ALLOW"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"
    DENY = "DENY"


@dataclass(frozen=True)
class PolicyContext:
    driver_risk: RiskLevel
    driver_quality: QualityStatus
    vehicle_moving: bool
    user_intent: str


@dataclass(frozen=True)
class PolicyResult:
    decision: PolicyDecision
    reason: str
    risk: ActionRisk | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskPolicy:
    high_driver_risk: RiskLevel
    fatigue_warning: str
    unknown_driver_warning: str


@dataclass(frozen=True)
class RecommendationPolicy:
    enabled: bool = False
    cooldown_ms: int = 60000


_FATIGUE_WARNING = "检测到高疲劳风险，请优先安全停车休息；音乐不能替代休息。"
_UNKNOWN_DRIVER_WARNING = "驾驶员风险状态未知或不可用，无法依据当前观测判断风险。"


@dataclass(frozen=True)
class AgentPolicy:
    tool_risks: dict[str, ActionRisk]
    risk_policy: RiskPolicy
    recommendation: RecommendationPolicy = RecommendationPolicy()

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> AgentPolicy:
        config_path = (
            Path(path)
            if path is not None
            else (Path(__file__).parents[2] / "config" / "agent_policy.yaml")
        )
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or not {
                "schema_version",
                "tool_risks",
                "risk_policy",
            }.issubset(data)
            or set(data)
            - {
                "schema_version",
                "tool_risks",
                "risk_policy",
                "recommendation",
            }
        ):
            raise ValueError("agent policy configuration has invalid fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 2:
            raise ValueError("unsupported agent policy schema version")

        raw_risks = data["tool_risks"]
        if not isinstance(raw_risks, dict) or not raw_risks:
            raise ValueError("tool_risks must be a nonempty mapping")
        risks: dict[str, ActionRisk] = {}
        for name, value in raw_risks.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                raise ValueError("tool risk entries must have string names and values")
            try:
                risks[name] = ActionRisk(value)
            except ValueError as error:
                raise ValueError(f"invalid risk for tool {name}: {value}") from error

        raw_policy = data["risk_policy"]
        if not isinstance(raw_policy, dict) or set(raw_policy) != {
            "high_driver_risk",
            "fatigue_warning_code",
            "unknown_driver_warning_code",
        }:
            raise ValueError("risk_policy has invalid fields")
        if raw_policy["high_driver_risk"] != RiskLevel.HIGH:
            raise ValueError("high_driver_risk must be HIGH")
        if raw_policy["fatigue_warning_code"] != "REST_REQUIRED":
            raise ValueError("unsupported fatigue warning code")
        if raw_policy["unknown_driver_warning_code"] != "STATE_UNAVAILABLE":
            raise ValueError("unsupported unknown driver warning code")
        raw_recommendation = data.get("recommendation")
        if raw_recommendation is None and "recommendation" in data:
            raise ValueError("recommendation must be a mapping")
        if raw_recommendation is not None:
            if not isinstance(raw_recommendation, dict) or set(raw_recommendation) != {
                "enabled",
                "cooldown_ms",
            }:
                raise ValueError("recommendation has invalid fields")
            if type(raw_recommendation["enabled"]) is not bool:
                raise ValueError("recommendation.enabled must be boolean")
            if (
                type(raw_recommendation["cooldown_ms"]) is not int
                or raw_recommendation["cooldown_ms"] <= 0
            ):
                raise ValueError("recommendation.cooldown_ms must be positive")
        return cls(
            tool_risks=risks,
            risk_policy=RiskPolicy(
                high_driver_risk=RiskLevel.HIGH,
                fatigue_warning=_FATIGUE_WARNING,
                unknown_driver_warning=_UNKNOWN_DRIVER_WARNING,
            ),
            recommendation=(
                RecommendationPolicy(**raw_recommendation)
                if raw_recommendation is not None
                else RecommendationPolicy()
            ),
        )

    def evaluate(self, tool: ToolDefinition, context: PolicyContext) -> PolicyResult:
        risk = self.tool_risks.get(tool.name)
        if risk is None:
            return PolicyResult(PolicyDecision.DENY, "Tool is not configured.", None)
        if tool.read_only != (risk is ActionRisk.READ_ONLY):
            return PolicyResult(
                PolicyDecision.DENY,
                "Configured risk conflicts with the tool's read-only flag.",
                risk,
            )
        if tool.requires_confirmation != (risk is ActionRisk.CONFIRMATION_REQUIRED):
            return PolicyResult(
                PolicyDecision.DENY,
                "Configured risk conflicts with the tool's confirmation requirement.",
                risk,
            )

        warnings: list[str] = []
        driver_known = (
            context.driver_quality is QualityStatus.KNOWN
            and context.driver_risk is not RiskLevel.UNKNOWN
        )
        if not driver_known:
            warnings.append(self.risk_policy.unknown_driver_warning)
        elif context.driver_risk is self.risk_policy.high_driver_risk:
            warnings.append(self.risk_policy.fatigue_warning)

        if risk is ActionRisk.CONFIRMATION_REQUIRED:
            reason = "This action requires explicit confirmation."
            if context.vehicle_moving:
                reason += " Vehicle is moving."
            return PolicyResult(
                PolicyDecision.REQUIRE_CONFIRMATION, reason, risk, tuple(warnings)
            )
        if risk is ActionRisk.READ_ONLY:
            return PolicyResult(
                PolicyDecision.ALLOW,
                "Read-only action is allowed.",
                risk,
                tuple(warnings),
            )
        return PolicyResult(
            PolicyDecision.ALLOW, "Reversible action is allowed.", risk, tuple(warnings)
        )

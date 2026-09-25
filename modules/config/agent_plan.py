"""Validated limits for the bounded rest-stop task plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AgentPlanConfig:
    max_steps: int = 9
    max_recoveries: int = 1
    max_candidates: int = 3

    @classmethod
    def load(cls, path: Path | str | None = None) -> AgentPlanConfig:
        config_path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("agent_plan.yaml")
        )
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "max_steps",
            "max_recoveries",
            "max_candidates",
        }:
            raise ValueError("agent plan configuration has invalid fields")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("unsupported agent plan schema version")
        for name in ("max_steps", "max_recoveries", "max_candidates"):
            if type(raw[name]) is not int or raw[name] < 1:
                raise ValueError(f"agent plan {name} must be a positive integer")
        if raw["max_recoveries"] >= raw["max_steps"]:
            raise ValueError("agent plan recovery budget must be less than step budget")
        return cls(
            max_steps=raw["max_steps"],
            max_recoveries=raw["max_recoveries"],
            max_candidates=raw["max_candidates"],
        )

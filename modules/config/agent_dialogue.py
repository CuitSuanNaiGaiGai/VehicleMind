"""Validated feature limits for bounded agent dialogue state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AgentDialogueConfig:
    enabled: bool = False
    candidate_ttl_seconds: float = 120.0
    max_revisions: int = 3
    interpreter_timeout_seconds: float = 15.0

    @classmethod
    def load(cls, path: Path | str | None = None) -> AgentDialogueConfig:
        config_path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("agent_dialogue.yaml")
        )
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "enabled",
            "candidate_ttl_seconds",
            "max_revisions",
            "interpreter_timeout_seconds",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("agent dialogue configuration has invalid fields")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("unsupported agent dialogue schema version")
        if type(raw["enabled"]) is not bool:
            raise ValueError("agent dialogue enabled must be a boolean")
        if type(raw["max_revisions"]) is not int or not 1 <= raw["max_revisions"] <= 3:
            raise ValueError(
                "agent dialogue max_revisions must be an integer from 1 to 3"
            )

        for name, maximum in (
            ("candidate_ttl_seconds", 120.0),
            ("interpreter_timeout_seconds", 15.0),
        ):
            value = raw[name]
            if type(value) not in {int, float} or (
                type(value) is float and not math.isfinite(value)
            ):
                raise ValueError(f"agent dialogue {name} must be a finite number")
            if not 0 < value <= maximum:
                raise ValueError(f"agent dialogue {name} is outside its allowed range")

        return cls(
            enabled=raw["enabled"],
            candidate_ttl_seconds=float(raw["candidate_ttl_seconds"]),
            max_revisions=raw["max_revisions"],
            interpreter_timeout_seconds=float(raw["interpreter_timeout_seconds"]),
        )

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace

from modules.config.perception import PerceptionConfig


ALLOWED_DOMAINS = {"phone", "driving", "lane"}


def resolve_overrides(
    config: PerceptionConfig,
    overrides: Mapping[str, object],
) -> PerceptionConfig:
    """Apply validated dotted-path overrides to immutable perception config."""

    resolved = config
    for path, value in overrides.items():
        domain, separator, field_name = path.partition(".")
        if not separator or domain not in ALLOWED_DOMAINS:
            raise ValueError(f"unknown perception override: {path}")

        section = getattr(resolved, domain)
        if field_name not in {item.name for item in fields(section)}:
            raise ValueError(f"unknown perception override: {path}")

        updated_section = replace(section, **{field_name: value})
        resolved = replace(resolved, **{domain: updated_section})
    return resolved

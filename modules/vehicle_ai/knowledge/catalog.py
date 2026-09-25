"""Validate source provenance and content before it enters an index."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from modules.vehicle_ai.knowledge.models import KnowledgeSource

_FIELDS = (
    "source_id",
    "title",
    "source_uri",
    "section",
    "version",
    "published_at",
    "profile",
    "topic",
    "text_path",
    "sha256",
    "license_note",
    "status",
)
_PROFILES = frozenset({"vehicle_common", "vehiclemind_demo"})
_SOURCE_ID = re.compile(r"K[0-9]{3,}")
_HASH = re.compile(r"[0-9a-f]{64}")


class KnowledgeCatalog:
    def load(self, manifest_path: Path) -> tuple[KnowledgeSource, ...]:
        root = manifest_path.resolve().parent
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
            raise ValueError("source catalog must contain a sources list")
        sources = tuple(self._parse_source(item, root) for item in raw["sources"])
        self.validate(sources)
        return sources

    def validate(self, sources: Sequence[KnowledgeSource]) -> None:
        seen: set[str] = set()
        for source in sources:
            if source.source_id in seen:
                raise ValueError(f"duplicate source_id: {source.source_id}")
            seen.add(source.source_id)
            if source.profile not in _PROFILES:
                raise ValueError(f"invalid profile: {source.profile}")
            if hashlib.sha256(source.text.encode("utf-8")).hexdigest() != source.sha256:
                raise ValueError(f"sha256 mismatch for {source.source_id}")

    @staticmethod
    def _parse_source(item: Any, root: Path) -> KnowledgeSource:
        if not isinstance(item, dict):
            raise ValueError("source entry must be a mapping")
        values: dict[str, str] = {}
        for field in _FIELDS:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing or invalid {field}")
            values[field] = value.strip()
        if set(item) != set(_FIELDS):
            raise ValueError("source entry has unknown fields")
        if not _SOURCE_ID.fullmatch(values["source_id"]):
            raise ValueError("invalid source_id")
        if values["profile"] not in _PROFILES:
            raise ValueError("invalid profile")
        if not _HASH.fullmatch(values["sha256"]):
            raise ValueError("invalid sha256")
        if values["status"] != "active":
            raise ValueError("only active sources may enter the catalog")
        relative = Path(values["text_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("text_path must stay inside the catalog")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"invalid text_path: {relative}")
        return KnowledgeSource(**values, text=path.read_text(encoding="utf-8"))

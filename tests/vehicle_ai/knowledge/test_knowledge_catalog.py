"""Contract tests for source provenance and catalog boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog


def _source(text: str, **overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "source_id": "K001",
        "title": "驾驶提醒",
        "source_uri": "https://example.org/guide#fatigue",
        "section": "疲劳提醒",
        "version": "2026-09",
        "published_at": "2026-09-01",
        "profile": "vehicle_common",
        "topic": "driver_safety",
        "text_path": "common/fatigue.md",
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "license_note": "项目原创文本",
        "status": "active",
    }
    source.update(overrides)
    return source


def _manifest(tmp_path: Path, sources: list[dict[str, object]], text: str) -> Path:
    content = tmp_path / "common" / "fatigue.md"
    content.parent.mkdir(parents=True)
    content.write_text(text, encoding="utf-8")
    manifest = tmp_path / "source_catalog.yaml"
    manifest.write_text(yaml.safe_dump({"sources": sources}), encoding="utf-8")
    return manifest


def test_catalog_loads_traced_source(tmp_path: Path) -> None:
    text = "疲劳时应及时停车休息。"
    manifest = _manifest(tmp_path, [_source(text)], text)

    (source,) = KnowledgeCatalog().load(manifest)

    assert source.source_id == "K001"
    assert source.profile == "vehicle_common"
    assert source.text == text
    assert source.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("missing", ["title", "source_uri", "section", "license_note"])
def test_catalog_rejects_missing_provenance(tmp_path: Path, missing: str) -> None:
    text = "测试正文"
    source = _source(text)
    del source[missing]
    manifest = _manifest(tmp_path, [source], text)

    with pytest.raises(ValueError, match=missing):
        KnowledgeCatalog().load(manifest)


def test_catalog_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    text = "测试正文"
    manifest = _manifest(tmp_path, [_source(text), _source(text)], text)

    with pytest.raises(ValueError, match="duplicate source_id"):
        KnowledgeCatalog().load(manifest)


def test_catalog_rejects_wrong_hash(tmp_path: Path) -> None:
    text = "测试正文"
    manifest = _manifest(tmp_path, [_source(text, sha256="0" * 64)], text)

    with pytest.raises(ValueError, match="sha256"):
        KnowledgeCatalog().load(manifest)


def test_catalog_rejects_missing_text(tmp_path: Path) -> None:
    text = "测试正文"
    manifest = _manifest(tmp_path, [_source(text, text_path="common/missing.md")], text)

    with pytest.raises(ValueError, match="text_path"):
        KnowledgeCatalog().load(manifest)


@pytest.mark.parametrize("profile", ["another_car", "", "https://evil.example"])
def test_catalog_rejects_unapproved_profile(tmp_path: Path, profile: str) -> None:
    text = "测试正文"
    manifest = _manifest(tmp_path, [_source(text, profile=profile)], text)

    with pytest.raises(ValueError, match="profile"):
        KnowledgeCatalog().load(manifest)


@pytest.mark.parametrize("text_path", ["../outside.md", "/etc/passwd"])
def test_catalog_rejects_text_outside_catalog(tmp_path: Path, text_path: str) -> None:
    text = "测试正文"
    manifest = _manifest(tmp_path, [_source(text, text_path=text_path)], text)

    with pytest.raises(ValueError, match="text_path"):
        KnowledgeCatalog().load(manifest)

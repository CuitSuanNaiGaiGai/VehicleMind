"""Index builds use scoped corpora and publish only complete generations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog
from modules.vehicle_ai.knowledge.index_builder import (
    IndexBuilder,
    publish_staged_index,
    sources_for_profile,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCES = KnowledgeCatalog().load(ROOT / "config/knowledge/source_catalog.yaml")


class FakeDocumentClient:
    def __init__(self, *, final_status: str = "processed") -> None:
        self.filenames: list[str] = []
        self.final_status = final_status

    def upload(self, filename: str, content: bytes) -> str:
        assert content
        self.filenames.append(filename)
        return f"track-{len(self.filenames)}"

    def status(self, track_id: str) -> str:
        assert track_id.startswith("track-")
        return self.final_status


def test_common_index_contains_only_common_sources(tmp_path: Path) -> None:
    client = FakeDocumentClient()
    builder = IndexBuilder(
        client, tmp_path / "stage", poll_seconds=0, timeout_seconds=1
    )

    result = builder.build("vehicle_common", SOURCES)

    assert result.profile == "vehicle_common"
    assert result.source_count == 10
    assert len(client.filenames) == 10
    assert all(
        filename.startswith(f"K{number:03d}-")
        for number, filename in enumerate(client.filenames, 1)
    )
    manifest = json.loads((tmp_path / "stage/index_manifest.json").read_text())
    assert manifest["source_ids"] == [f"K{number:03d}" for number in range(1, 11)]


def test_demo_index_copies_common_sources_with_same_ids_and_hashes(
    tmp_path: Path,
) -> None:
    selected = sources_for_profile(SOURCES, "vehiclemind_demo")
    client = FakeDocumentClient()
    result = IndexBuilder(
        client, tmp_path / "stage", poll_seconds=0, timeout_seconds=1
    ).build("vehiclemind_demo", SOURCES)

    assert result.source_count == 20
    assert len(client.filenames) == 20
    assert [(item.source_id, item.sha256) for item in selected[:10]] == [
        (item.source_id, item.sha256) for item in SOURCES[:10]
    ]


def test_failed_upload_does_not_replace_existing_active_index(tmp_path: Path) -> None:
    active = tmp_path / "active"
    active.mkdir()
    (active / "index_manifest.json").write_text("old complete index")
    client = FakeDocumentClient(final_status="failed")

    with pytest.raises(RuntimeError, match="failed"):
        IndexBuilder(
            client, tmp_path / "stage", poll_seconds=0, timeout_seconds=1
        ).build("vehicle_common", SOURCES)

    assert (active / "index_manifest.json").read_text() == "old complete index"
    assert not (tmp_path / "stage/index_manifest.json").exists()


def test_rejected_upload_never_publishes_manifest(tmp_path: Path) -> None:
    class RejectedClient(FakeDocumentClient):
        def upload(self, filename: str, content: bytes) -> str:
            raise RuntimeError("upload rejected")

    with pytest.raises(RuntimeError, match="rejected"):
        IndexBuilder(RejectedClient(), tmp_path / "stage").build("vehicle_common", SOURCES)
    assert not (tmp_path / "stage/index_manifest.json").exists()


def test_unknown_profile_cannot_build_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        IndexBuilder(FakeDocumentClient(), tmp_path / "stage").build("unexpected", SOURCES)


def test_pending_upload_times_out_without_publishing(tmp_path: Path) -> None:
    client = FakeDocumentClient(final_status="pending")

    with pytest.raises(TimeoutError, match="index"):
        IndexBuilder(
            client, tmp_path / "stage", poll_seconds=0, timeout_seconds=0
        ).build("vehicle_common", SOURCES)

    assert not (tmp_path / "stage/index_manifest.json").exists()


def test_staged_storage_swap_keeps_recoverable_old_generation(tmp_path: Path) -> None:
    active = tmp_path / "active"
    (active / "storage").mkdir(parents=True)
    (active / "storage/old.txt").write_text("old")
    staged = tmp_path / "stage"
    (staged / "storage").mkdir(parents=True)
    (staged / "inputs").mkdir()
    (staged / "storage/new.txt").write_text("new")
    (staged / "storage/index_manifest.json").write_text("new complete index")
    backup = tmp_path / "backup"

    publish_staged_index(staged, active, backup)

    assert (active / "storage/new.txt").read_text() == "new"
    assert (active / "inputs").is_dir()
    assert (backup / "storage/old.txt").read_text() == "old"


def test_publish_refuses_missing_staged_storage(tmp_path: Path) -> None:
    active = tmp_path / "active"
    (active / "storage").mkdir(parents=True)
    (active / "storage/old.txt").write_text("old")

    with pytest.raises(ValueError, match="staged"):
        publish_staged_index(tmp_path / "missing", active, tmp_path / "backup")

    assert (active / "storage/old.txt").read_text() == "old"

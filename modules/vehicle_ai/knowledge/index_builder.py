"""Build a scoped LightRAG generation before swapping it into service."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog
from modules.vehicle_ai.knowledge.models import KnowledgeSource

LIGHTRAG_VERSION = "1.5.7"
PROFILES = frozenset({"vehicle_common", "vehiclemind_demo"})


class DocumentClient(Protocol):
    def upload(self, filename: str, content: bytes) -> str: ...

    def status(self, track_id: str) -> str: ...


@dataclass(frozen=True)
class IndexBuildResult:
    profile: str
    source_count: int
    manifest_sha256: str
    indexed_at: str
    lightrag_version: str
    success: bool


def sources_for_profile(
    sources: Sequence[KnowledgeSource], profile: str
) -> tuple[KnowledgeSource, ...]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    return tuple(
        source
        for source in sources
        if source.profile == "vehicle_common" or source.profile == profile
    )


class LightRAGDocumentClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def upload(self, filename: str, content: bytes) -> str:
        boundary = "vehiclemind-knowledge-upload"
        body = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                "Content-Type: text/markdown\r\n\r\n"
            ).encode("utf-8")
            + content
            + f"\r\n--{boundary}--\r\n".encode()
        )
        request = Request(
            f"{self.base_url}/documents/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)
        if payload.get("status") != "success" or not payload.get("track_id"):
            raise RuntimeError(f"LightRAG upload failed for {filename}")
        return str(payload["track_id"])

    def status(self, track_id: str) -> str:
        with urlopen(
            f"{self.base_url}/documents/track_status/{track_id}",
            timeout=self.timeout_seconds,
        ) as response:
            payload = json.load(response)
        documents = payload.get("documents")
        if not isinstance(documents, list) or not documents:
            return "pending"
        statuses = {item.get("status") for item in documents if isinstance(item, dict)}
        if "failed" in statuses:
            return "failed"
        if statuses == {"processed"}:
            return "processed"
        return "pending"


class IndexBuilder:
    def __init__(
        self,
        client: DocumentClient,
        staging_dir: Path,
        *,
        poll_seconds: float = 3,
        timeout_seconds: float = 600,
    ) -> None:
        self.client = client
        self.staging_dir = staging_dir
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def build(
        self, profile: str, sources: Sequence[KnowledgeSource]
    ) -> IndexBuildResult:
        KnowledgeCatalog().validate(sources)
        selected = sources_for_profile(sources, profile)
        if not selected:
            raise ValueError(f"no sources for {profile}")
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        tracks = {
            source.source_id: self.client.upload(
                Path(source.text_path).name, source.text.encode("utf-8")
            )
            for source in selected
        }
        deadline = time.monotonic() + self.timeout_seconds
        remaining = dict(tracks)
        while remaining:
            for source_id, track_id in tuple(remaining.items()):
                status = self.client.status(track_id)
                if status == "failed":
                    raise RuntimeError(f"indexing failed for {source_id}")
                if status == "processed":
                    del remaining[source_id]
            if remaining:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"index build timed out for {profile}")
                time.sleep(self.poll_seconds)
        identity = [
            {"source_id": source.source_id, "sha256": source.sha256}
            for source in selected
        ]
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        result = IndexBuildResult(
            profile=profile,
            source_count=len(selected),
            manifest_sha256=digest,
            indexed_at=datetime.now(UTC).isoformat(),
            lightrag_version=LIGHTRAG_VERSION,
            success=True,
        )
        manifest = {
            **result.__dict__,
            "source_ids": [source.source_id for source in selected],
            "sources": identity,
        }
        temporary = self.staging_dir / "index_manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.staging_dir / "index_manifest.json")
        return result


def publish_staged_index(staged: Path, active: Path, backup: Path) -> None:
    if not (
        staged.is_dir()
        and (staged / "storage/index_manifest.json").is_file()
        and (staged / "inputs").is_dir()
    ):
        raise ValueError("staged index lacks a complete manifest")
    if backup.exists():
        raise ValueError("backup directory already exists")
    active.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)
    had_active = active.exists()
    if had_active:
        active.replace(backup)
    try:
        staged.replace(active)
    except Exception:
        if had_active:
            backup.replace(active)
        raise

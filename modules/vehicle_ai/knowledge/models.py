"""Small immutable types shared by the knowledge pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    title: str
    source_uri: str
    section: str
    version: str
    published_at: str
    profile: str
    topic: str
    text_path: str
    sha256: str
    license_note: str
    status: str
    text: str


@dataclass(frozen=True)
class KnowledgeReference:
    source_id: str
    title: str
    section: str
    source_uri: str
    profile: str


@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    text: str
    rank: int
    reference: KnowledgeReference


@dataclass(frozen=True)
class RetrievalResult:
    profile: str
    query: str
    chunks: tuple[RetrievedChunk, ...]
    latency_ms: float
    request_id: str
    error_code: str | None = None
    error_message: str | None = None

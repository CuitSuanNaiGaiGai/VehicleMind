"""Frozen AI-assisted internal RAG cases, separate from human gold labels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RagCase:
    case_id: str
    query: str
    profile: str
    answerable: bool
    expected_source_ids: tuple[str, ...]
    expected_abstention_reason: str | None = None


def load_cases(path: Path) -> tuple[tuple[RagCase, ...], str]:
    content = path.read_bytes()
    raw = yaml.safe_load(content)
    if not isinstance(raw, dict) or raw.get("version") != "rag-internal-v1":
        raise ValueError("unsupported RAG evaluation version")
    if raw.get("review_status") != "ai_assisted_internal":
        raise ValueError("RAG cases must disclose AI-assisted review")
    items = raw.get("cases")
    if not isinstance(items, list):
        raise ValueError("RAG cases must be a list")
    cases = tuple(_parse_case(item) for item in items)
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)) or ids != [f"R{index:02d}" for index in range(1, 31)]:
        raise ValueError("RAG case IDs must be unique R01-R30")
    if (
        sum(case.answerable for case in cases) != 20
        or sum(case.expected_abstention_reason == "no_answer" for case in cases) != 5
        or sum(case.expected_abstention_reason == "profile_mismatch" for case in cases)
        != 5
    ):
        raise ValueError("RAG cases require 20 answerable, 5 no-answer, 5 mismatch")
    return cases, hashlib.sha256(content).hexdigest()


def _parse_case(item: Any) -> RagCase:
    if not isinstance(item, dict):
        raise ValueError("RAG case must be a mapping")
    required = {"case_id", "query", "profile", "answerable", "expected_source_ids"}
    if not required.issubset(item) or set(item) - (
        required | {"expected_abstention_reason"}
    ):
        raise ValueError("RAG case fields mismatch")
    if not all(
        isinstance(item[key], str) and item[key] for key in ("case_id", "query")
    ):
        raise ValueError("RAG case ID/query must be text")
    if item["profile"] not in {"vehicle_common", "vehiclemind_demo"}:
        raise ValueError("unknown RAG case profile")
    if type(item["answerable"]) is not bool:
        raise ValueError("answerable must be boolean")
    sources = item["expected_source_ids"]
    if not isinstance(sources, list) or not all(
        isinstance(source, str) and source.startswith("K") for source in sources
    ):
        raise ValueError("expected_source_ids must be a list of IDs")
    reason = item.get("expected_abstention_reason")
    if item["answerable"]:
        if not sources or reason is not None:
            raise ValueError("answerable case requires source IDs without abstention")
    elif reason not in {"no_answer", "profile_mismatch"}:
        raise ValueError("unanswerable case requires a reason")
    return RagCase(
        item["case_id"],
        item["query"],
        item["profile"],
        item["answerable"],
        tuple(sources),
        reason,
    )

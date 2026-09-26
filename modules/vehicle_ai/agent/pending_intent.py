from __future__ import annotations

import re
from typing import Literal

from modules.vehicle_ai.agent.target_resolution import resolve_target


PendingIntent = Literal["reject", "change_target"]

_TARGET_CLAUSE_SEPARATOR = re.compile(r"[，,；;。！？!?\n]")
_ALTERNATIVE_DESTINATION = re.compile(
    r"(?:或者|还是|或|\bor\b|\binstead\b|\brather\b)", re.IGNORECASE
)
_SEARCH_CONTINUATION = re.compile(
    r"^(?:"
    r"请重新搜索|重新搜索|再搜索|请搜索|搜索一下|搜索|"
    r"请重新查找|重新查找|重新查询|"
    r"找不到就不要导航|找不到时不要导航|没有结果就不要导航|"
    r"不要导航|先别导航|无需导航|"
    r"please\s+search|search(?:\s+again)?|"
    r"if\s+not\s+found(?:,?\s+do\s+not\s+navigate)?|"
    r"if\s+none(?:,?\s+do\s+not\s+navigate)?|"
    r"do\s+not\s+navigate|don't\s+navigate"
    r")",
    re.IGNORECASE,
)


def classify_pending_intent(text: str) -> PendingIntent | None:
    normalized = text.strip().rstrip("。.!！?？").strip().casefold()
    if normalized in {"取消", "不要了", "cancel", "no thanks"}:
        return "reject"
    for prefix in ("换成", "改去", "instead ", "change destination to "):
        if normalized.startswith(prefix) and normalized[len(prefix) :].strip():
            return "change_target"
    return None


def requested_target(text: str) -> str | None:
    cleaned = text.strip().rstrip("。.!！?？").strip()
    normalized = cleaned.casefold()
    for prefix in ("换成", "改去", "instead ", "change destination to "):
        if normalized.startswith(prefix):
            body = cleaned[len(prefix) :].strip()
            if not body:
                return None
            parts = _TARGET_CLAUSE_SEPARATOR.split(body, maxsplit=1)
            first_clause = parts[0].strip()
            if len(parts) == 1:
                return first_clause or None
            continuation = parts[1].strip()
            if (
                first_clause
                and continuation
                and not _ALTERNATIVE_DESTINATION.search(body)
                and _SEARCH_CONTINUATION.match(continuation)
            ):
                return first_clause
            return body
    return None


def search_result_matches_target(target: str, result: dict) -> bool:
    return resolve_target(target, [result]).status == "matched"

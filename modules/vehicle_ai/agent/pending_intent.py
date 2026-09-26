from __future__ import annotations

import re
from typing import Literal

from modules.vehicle_ai.agent.target_resolution import resolve_target


PendingIntent = Literal["reject", "change_target"]

_TARGET_CLAUSE_SEPARATOR = re.compile(r"[，,；;。！？!?\n]")
_SEARCH_CONTINUATION = re.compile(
    r"(?:"
    r"帮我重新搜索|请重新搜索|重新搜索|再搜索|请搜索|搜索一下|搜索|"
    r"请重新查找|重新查找|重新查询|"
    r"如果找不到就不要导航|如果找不到时不要导航|"
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
            parts = [part.strip() for part in _TARGET_CLAUSE_SEPARATOR.split(body)]
            first_clause = parts[0]
            continuations = parts[1:]
            if (
                first_clause
                and continuations
                and all(
                    continuation and _SEARCH_CONTINUATION.fullmatch(continuation)
                    for continuation in continuations
                )
            ):
                return first_clause
            return first_clause if not continuations else body
    return None


def search_result_matches_target(target: str, result: dict) -> bool:
    return resolve_target(target, [result]).status == "matched"

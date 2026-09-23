from __future__ import annotations

from typing import Literal


PendingIntent = Literal["reject", "change_target"]


def classify_pending_intent(text: str) -> PendingIntent | None:
    normalized = text.strip().rstrip("。.!！?？").strip().casefold()
    if normalized in {"取消", "不要了", "cancel", "no thanks"}:
        return "reject"
    for prefix in ("换成", "改去", "instead ", "change destination to "):
        if normalized.startswith(prefix) and normalized[len(prefix) :].strip():
            return "change_target"
    return None


def requested_target(text: str) -> str | None:
    normalized = text.strip().rstrip("。.!！?？").strip().casefold()
    for prefix in ("换成", "改去", "instead ", "change destination to "):
        if normalized.startswith(prefix):
            target = normalized[len(prefix) :].strip()
            return target or None
    return None


def search_result_matches_target(target: str, result: dict) -> bool:
    names = (result.get("name"), *result.get("aliases", ()))
    return any(
        isinstance(name, str) and name.casefold().strip() == target for name in names
    )

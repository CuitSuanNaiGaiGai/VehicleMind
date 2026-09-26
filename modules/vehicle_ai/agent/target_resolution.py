"""Resolve explicit destination requests against the current tool candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TargetResolutionStatus = Literal["matched", "not_found", "ambiguous"]


@dataclass(frozen=True)
class TargetResolution:
    """The exact current candidate match for a user-requested destination."""

    status: TargetResolutionStatus
    target: str
    candidate: dict[str, Any] | None
    candidate_ids: tuple[str, ...]


def _normalize_label(value: str) -> str:
    return value.strip().casefold()


def _labels_for(candidate: dict[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    for key in ("poi_id", "name", "display_name_zh"):
        value = candidate.get(key)
        if isinstance(value, str):
            labels.append(value)

    aliases = candidate.get("aliases")
    if isinstance(aliases, (list, tuple)):
        labels.extend(alias for alias in aliases if isinstance(alias, str))
    return tuple(labels)


def resolve_target(target: str, candidates: list[dict[str, Any]]) -> TargetResolution:
    """Match a normalized exact label and reject matches across distinct IDs."""

    normalized_target = _normalize_label(target)
    matched_by_id: dict[str, dict[str, Any]] = {}
    if normalized_target:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = candidate.get("poi_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                continue
            if any(
                _normalize_label(label) == normalized_target
                for label in _labels_for(candidate)
            ):
                matched_by_id.setdefault(candidate_id, candidate)

    candidate_ids = tuple(matched_by_id)
    if not candidate_ids:
        return TargetResolution(
            status="not_found",
            target=target.strip(),
            candidate=None,
            candidate_ids=(),
        )
    if len(candidate_ids) > 1:
        return TargetResolution(
            status="ambiguous",
            target=target.strip(),
            candidate=None,
            candidate_ids=candidate_ids,
        )
    return TargetResolution(
        status="matched",
        target=target.strip(),
        candidate=matched_by_id[candidate_ids[0]],
        candidate_ids=candidate_ids,
    )


def _display_label(candidate: dict[str, Any]) -> str | None:
    for key in ("display_name_zh", "name", "poi_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def target_resolution_message(
    resolution: TargetResolution, *, pending_created: bool
) -> str:
    """Return finite Chinese feedback based only on the resolution and pending."""

    target = resolution.target or "所选地点"
    if resolution.status == "ambiguous":
        return (
            f"当前搜索候选中有多个地点与“{target}”完全匹配，"
            "未创建待确认导航。请从候选中提供更明确的名称。"
        )
    if resolution.status == "not_found":
        return (
            f"当前搜索候选中没有与“{target}”完全匹配的地点，"
            "未创建待确认导航。请提供候选中的准确名称，或重新搜索。"
        )

    candidate = resolution.candidate
    display_label = _display_label(candidate) if candidate is not None else None
    if display_label is None:
        display_label = target
        pending_created = False
    if pending_created:
        return (
            f"已按“{target}”匹配到本次搜索候选“{display_label}”，"
            "新的导航操作已待确认；"
            "请确认后再导航。"
        )
    return (
        f"已按“{target}”匹配到本次搜索候选“{display_label}”，"
        "但未能创建新的待确认导航操作，"
        "因此当前不能导航。"
    )


def target_search_failure_message(error: str | None) -> str:
    """Describe a failed target search without converting it to a no-match."""

    messages = {
        "NO_RESULTS": "本次地点搜索没有返回可用候选，未创建待确认导航。",
        "TRANSIENT_ERROR": "本次地点搜索暂时失败，未创建待确认导航。",
        "INVALID_SEARCH_FILTER": "本次地点搜索条件无效，未创建待确认导航。",
        "TARGET_SEARCH_NOT_COMPLETED": "本次地点搜索尚未得到可核验的匹配结果，未创建待确认导航。",
    }
    if error is None:
        return "本次地点搜索未能完成，未创建待确认导航。"
    return messages.get(error, f"本次地点搜索失败（{error}），未创建待确认导航。")

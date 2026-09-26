"""Render only decision briefs recorded in the actual model requests."""

from __future__ import annotations

import json
from html import escape
from typing import Any


_CONTEXT_MARKER = "CURRENT RELEVANT VEHICLE CONTEXT:\n"
_BRIEF_MARKER = "\n\nDECISION BRIEF:\n"
_TARGET_STATUS_LABELS = {
    "matched": "唯一匹配",
    "not_found": "未找到",
    "ambiguous": "多个匹配",
}


def _request_brief(request: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    decoder = json.JSONDecoder()
    for message in request.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(_CONTEXT_MARKER):
            continue
        context_text = content[len(_CONTEXT_MARKER) :]
        try:
            _, context_end = decoder.raw_decode(context_text)
        except ValueError:
            continue
        brief_text = context_text[context_end:]
        if not brief_text.startswith(_BRIEF_MARKER):
            continue
        try:
            brief, _ = decoder.raw_decode(brief_text[len(_BRIEF_MARKER) :])
        except ValueError:
            return "invalid", None
        if not isinstance(brief, dict):
            return "invalid", None
        return "recorded", brief
    return "missing", None


def _point_html(point: dict[str, Any]) -> str:
    code = escape(str(point.get("code", "未记录")))
    text = escape(str(point.get("text_zh", "未记录说明")))
    fields = point.get("evidence_fields", [])
    if not isinstance(fields, list):
        fields = []
    evidence = "、".join(escape(str(field)) for field in fields)
    evidence_html = f"<small>依据字段：{evidence}</small>" if evidence else ""
    return f"<li><code>{code}</code> · {text}{evidence_html}</li>"


def _brief_html(index: int, request: dict[str, Any]) -> str:
    state, brief = _request_brief(request)
    summary = f"第 {index} 次模型请求 · 决策简报"
    if state == "missing":
        body = "<p>本次请求未记录决策简报。</p>"
    elif state == "invalid":
        body = "<p>本次请求记录了简报标记，但内容无法解析。</p>"
    else:
        assert brief is not None
        points = brief.get("required_points", [])
        if not isinstance(points, list):
            points = []
        point_rows = (
            "".join(_point_html(point) for point in points if isinstance(point, dict))
            or "<li>未记录必需说明项。</li>"
        )
        unavailable = brief.get("unavailable_fields", {})
        if isinstance(unavailable, dict):
            unavailable_rows = "".join(
                f"<li><code>{escape(str(field))}</code> · {escape(str(status))}</li>"
                for field, status in unavailable.items()
            )
        else:
            unavailable_rows = ""
        if not unavailable_rows:
            unavailable_rows = "<li>未记录不可用字段。</li>"
        body = (
            f"<p>版本：{escape(str(brief.get('version', '未记录')))}</p>"
            f"<h4>应覆盖的依据</h4><ul>{point_rows}</ul>"
            f"<h4>不可用字段</h4><ul>{unavailable_rows}</ul>"
        )
    return f"<details><summary>{summary}</summary>{body}</details>"


def decision_brief_panel(requests: list[dict[str, Any]]) -> str:
    """Display one recorded or explicitly unrecorded brief per request."""
    rows = "".join(
        _brief_html(index, request)
        for index, request in enumerate(requests, 1)
        if isinstance(request, dict)
    )
    if not rows:
        rows = "<p>未记录模型请求。</p>"
    return f"<div class='decision-briefs'><h3>实际请求中的决策简报</h3>{rows}</div>"


def _matching_candidate_ids(event: dict[str, Any]) -> list[str]:
    candidate_ids = event.get("candidate_ids", [])
    if not isinstance(candidate_ids, (list, tuple)):
        return []
    return [
        candidate_id.strip()
        for candidate_id in candidate_ids
        if isinstance(candidate_id, str) and candidate_id.strip()
    ]


def _target_resolution_html(index: int, event: dict[str, Any]) -> str:
    target_value = event.get("target")
    target = target_value.strip() if isinstance(target_value, str) else ""
    target_html = escape(target or "未记录")
    status_value = event.get("status")
    status = status_value if isinstance(status_value, str) else ""
    status_label = _TARGET_STATUS_LABELS.get(status, "未识别")
    body = [f"<p>请求目标：{target_html}</p>", f"<p>解析结果：{status_label}</p>"]

    if status == "matched":
        name_value = event.get("selected_display_name")
        id_value = event.get("selected_id")
        name = name_value.strip() if isinstance(name_value, str) else ""
        candidate_id = id_value.strip() if isinstance(id_value, str) else ""
        complete_match = bool(name and candidate_id)
        if complete_match:
            body.append(
                "<p>规范候选："
                f"{escape(name)} · ID：<code>{escape(candidate_id)}</code></p>"
            )
        else:
            body.append("<p>匹配候选的规范名称或 ID 未记录完整。</p>")

        pending_value = event.get("pending_action_id")
        pending_id = pending_value.strip() if isinstance(pending_value, str) else ""
        if pending_id and complete_match:
            body.append(
                "<p>已记录新的待确认操作："
                f"<code>{escape(pending_id)}</code>；仍需用户确认后才能导航。</p>"
            )
        elif pending_id:
            body.append(
                "<p>虽记录了新的待确认操作，但候选信息不完整；"
                "当前不能确认目标或开始导航。</p>"
            )
        else:
            body.append("<p>没有新的待确认操作；当前不能导航。</p>")
    elif status == "ambiguous":
        candidate_ids = _matching_candidate_ids(event)
        if candidate_ids:
            ids_html = "、".join(
                f"<code>{escape(value)}</code>" for value in candidate_ids
            )
            body.append(f"<p>匹配候选 ID：{ids_html}</p>")
        else:
            body.append("<p>匹配候选 ID 未记录。</p>")
        body.append("<p>目标匹配不唯一，需要补充地点名称；未创建新的待确认导航。</p>")
    elif status == "not_found":
        body.append(
            "<p>未找到匹配候选：本次搜索中没有完全匹配项；未创建新的待确认导航。</p>"
        )
    else:
        body.append(
            "<p>解析状态未记录为可识别结果："
            f"<code>{escape(status or '未记录')}</code>；不能确认候选或导航状态。</p>"
        )

    source = event.get("source")
    quality = event.get("quality")
    if source == "search_nearby_rest_area" and quality == "TOOL_RESULT":
        body.append("<small>依据：本次地点搜索工具结果。</small>")
    else:
        body.append("<small>来源未能确认为本次地点搜索工具结果。</small>")
    return (
        f"<details><summary>第 {index} 次目标解析 · {target_html}</summary>"
        f"{''.join(body)}</details>"
    )


def target_resolution_panel(events: list[dict[str, Any]]) -> str:
    """Render recorded target-resolution events without inferring outcomes."""
    resolutions = [
        event
        for event in events
        if isinstance(event, dict) and event.get("kind") == "target_resolution"
    ]
    if not resolutions:
        return ""
    rows = "".join(
        _target_resolution_html(index, event)
        for index, event in enumerate(resolutions, 1)
    )
    return f"<div class='target-resolution'><h3>目标解析与新确认要求</h3>{rows}</div>"

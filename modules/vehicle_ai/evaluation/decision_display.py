"""Render only decision briefs recorded in the actual model requests."""

from __future__ import annotations

import json
from html import escape
from typing import Any


_CONTEXT_MARKER = "CURRENT RELEVANT VEHICLE CONTEXT:\n"
_BRIEF_MARKER = "\n\nDECISION BRIEF:\n"


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

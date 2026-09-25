"""Chinese task states and per-request evidence, drawn only from recorded data."""

import json
from html import escape


LABELS = {
    "IDLE": "空闲",
    "RUNNING": "执行中",
    "AWAITING_INPUT": "等待输入",
    "AWAITING_CONFIRMATION": "待确认",
    "COMPLETED": "完成",
    "FAILED": "失败",
    "CANCELLED": "已取消",
}
REASONS = {
    "MODEL_TIMEOUT": "模型请求超时",
    "MODEL_ERROR": "模型请求异常",
    "EMPTY_RESPONSE": "模型空回复",
    "TIME_BUDGET": "本轮时间预算耗尽",
    "TOOL_BUDGET": "工具次数预算耗尽",
    "ROUND_LIMIT": "模型轮次耗尽",
    "REPEATED_CALL": "重复工具调用已阻止",
    "INVALID_ARGUMENTS": "参数校验失败",
    "PENDING_EXPIRED": "待确认操作过期",
    "WRITE_OUTCOME_UNKNOWN": "写操作结果未知，等待核对",
    "TARGET_NOT_FOUND": "新目标未找到",
    "USER_CANCELLED": "用户取消",
    "CLARIFICATION_REQUESTED": "需要补充信息",
    "NO_REFERENT": "缺少可关联目标",
}


def task_panel(trial) -> str:
    rows = []
    for event in trial.agent_trace:
        if event["kind"] not in {
            "agent_reply",
            "confirmation",
            "rejection",
            "pending_expired",
        }:
            continue
        task = event["task"]
        status = task["status"]
        reason = task.get("reason")
        rows.append(
            f"<div class='turn'><span>{escape(status)}（{LABELS.get(status, '未知')}）</span>"
            f"<p>目标：{escape(task.get('goal', ''))}</p>"
            f"<p>原因：{escape(REASONS.get(reason, reason or '无'))}</p></div>"
        )
    for call in trial.tool_calls:
        if call.get("data", {}).get("outcome_unknown"):
            rows.append("<p>已阻止自动重试；已读取车机状态供人工核对。</p>")
    return (
        "<section class='card'><h2>任务进度与停止原因</h2>"
        + ("".join(rows) or "<p>旧报告未记录任务状态</p>")
        + "</section>"
    )


def evidence_panel(trial) -> str:
    rows = []
    for index, request in enumerate(trial.requests, 1):
        for message in request.get("messages", []):
            text = message.get("content") or ""
            marker = "CURRENT RELEVANT VEHICLE CONTEXT:\n"
            if not isinstance(text, str) or not text.startswith(marker):
                continue
            try:
                context, _ = json.JSONDecoder().raw_decode(text[len(marker) :])
            except ValueError:
                continue
            facts = []
            for topic, values in context.items():
                quality = values.get("quality_status", "UNKNOWN")
                facts.append(f"<p>{escape(topic)} · {escape(quality)}</p>")
                for name, source in values.get("field_evidence", {}).items():
                    facts.append(
                        f"<div class='fact'><span>{escape(name)} = {escape(str(values.get(name)))}</span>"
                        f"<strong>来源：{escape(str(source.get('source') or '未提供'))} · "
                        f"源时间：{escape(str(source.get('timestamp_ms')))} ms · "
                        f"接收年龄：{escape(str(round(source.get('age_seconds') or 0, 3)))} s · "
                        f"置信度：{escape(str(source.get('confidence')))}</strong></div>"
                    )
            rows.append(
                f"<details><summary>第 {index} 次模型请求 · 实际事实与来源</summary>{''.join(facts)}</details>"
            )
    return (
        "<section class='card'><h2>逐轮输入证据</h2><p class='note'>源时间可为离线视频时间，不等同于当前时刻；未提供置信度不代表确定。</p>"
        + "".join(rows)
        + "</section>"
    )

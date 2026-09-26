"""Chinese task states and per-request evidence, drawn only from recorded data."""

import json
from html import escape

from modules.vehicle_ai.evaluation.decision_display import decision_brief_panel


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
    "NO_RESULTS": "没有可行地点",
    "ALTERNATIVE_PENDING": "替代地点待重新确认",
    "PLAN_VERIFIED": "结果回读通过",
    "UNGROUNDED_POI_ID": "地点不在本次搜索候选中",
    "RECOVERY_BUDGET_EXCEEDED": "恢复次数达到上限",
}
PLAN_STATUS_LABELS = {
    "RUNNING": "进行中",
    "COMPLETED": "已完成",
    "FAILED": "失败",
    "CANCELLED": "已取消",
    "STOPPED_NO_RESULT": "无结果并停止",
}
PLAN_STEP_STATUS_LABELS = {
    "PENDING": "未开始",
    "RUNNING": "进行中",
    "SUCCEEDED": "已完成",
    "FAILED": "失败",
    "SKIPPED": "已跳过",
}
PLAN_STEP_LABELS = {
    "UNDERSTAND": "理解需求",
    "KNOWLEDGE": "查询知识",
    "SEARCH": "搜索地点",
    "SELECT": "选择候选",
    "AWAIT_CONFIRMATION": "等待确认",
    "EXECUTE": "执行导航",
    "VERIFY": "回读核验",
    "RECOVER": "失败恢复",
}


def _plan_panel(plan: dict) -> str:
    used_steps = len(plan.get("steps", []))
    max_steps = plan.get("max_steps", "未记录")
    remaining_steps = plan.get("remaining_step_budget")
    if isinstance(remaining_steps, int) and isinstance(max_steps, int):
        used_steps = max_steps - remaining_steps
    recoveries = plan.get("recovery_count", 0)
    max_recoveries = plan.get("max_recoveries", "未记录")
    state = plan.get("status", "未知")
    rows = []
    for index, step in enumerate(plan.get("steps", []), 1):
        name = str(step.get("name", "未知步骤"))
        status = str(step.get("status", "未知"))
        evidence = step.get("evidence_summary") or "暂无步骤摘要"
        error = step.get("error_code")
        error_html = f"<p>失败码：{escape(str(error))}</p>" if error else ""
        rows.append(
            "<div class='turn'><span>"
            f"{index}. {escape(PLAN_STEP_LABELS.get(name, name))} · "
            f"{escape(PLAN_STEP_STATUS_LABELS.get(status, status))}</span><p>{escape(evidence)}</p>"
            f"{error_html}</div>"
        )
    candidates = []
    for candidate in plan.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        display_name = str(
            candidate.get("display_name_zh") or candidate.get("name") or "休息地点"
        )
        poi_id = candidate.get("poi_id", "未记录")
        simulated = "模拟候选" if candidate.get("simulated", True) else "未标记模拟"
        candidates.append(
            f"<li>{escape(display_name)} · <code>{escape(poi_id)}</code> · {simulated}</li>"
        )
    selected = plan.get("selected_poi_id")
    return (
        "<details class='plan-details' open><summary>查看受限计划步骤</summary>"
        f"<p>计划状态：{escape(PLAN_STATUS_LABELS.get(state, state))} · "
        f"步骤预算：{escape(str(used_steps))} / {escape(str(max_steps))} · "
        f"恢复预算：{escape(str(recoveries))} / {escape(str(max_recoveries))}</p>"
        f"<p>已选择地点：<code>{escape(selected or '尚未选择')}</code></p>"
        f"{''.join(rows) or '<p>暂无步骤记录。</p>'}"
        f"{'<p>当前模拟候选</p><ul>' + ''.join(candidates) + '</ul>' if candidates else ''}"
        "</details>"
    )


def task_panel(trial) -> str:
    rows = []
    latest_plan = None
    for event in trial.agent_trace:
        if event["kind"] == "event_recommendation":
            label = (
                "确定性降级安全建议"
                if event.get("source") == "deterministic_fallback"
                else "事件主动建议"
            )
            rows.append(
                f"<div class='turn'><span>{label}</span>"
                f"<p>{escape(str(event.get('text') or '无建议文本'))}</p></div>"
            )
            continue
        if event["kind"] not in {
            "agent_reply",
            "confirmation",
            "rejection",
            "pending_expired",
        }:
            continue
        task = event["task"]
        if isinstance(task.get("plan"), dict):
            latest_plan = task["plan"]
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
    if latest_plan is None:
        for event in reversed(trial.agent_trace):
            plan = event.get("task", {}).get("plan")
            if isinstance(plan, dict):
                latest_plan = plan
                break
    plan_html = _plan_panel(latest_plan) if latest_plan is not None else ""
    return (
        "<section class='card'><h2>任务进度与停止原因</h2>"
        + ("".join(rows) or "<p>旧报告未记录任务状态</p>")
        + plan_html
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
        + decision_brief_panel(trial.requests)
        + "</section>"
    )

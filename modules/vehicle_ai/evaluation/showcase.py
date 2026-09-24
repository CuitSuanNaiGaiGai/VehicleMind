"""Human-readable projection of one recorded online Agent trial."""

from __future__ import annotations

import html
import json
from typing import Any

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import TrialResult


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _first_turn_observations(case: EvaluationCase) -> tuple[dict, dict, str]:
    cabin: dict[str, Any] = {}
    road: dict[str, Any] = {}
    user_text = "未提供"
    for step in case.steps:
        if "cabin" in step:
            cabin = step["cabin"]
        if "road" in step:
            road = step["road"]
        if "user_text" in step:
            user_text = step["user_text"]
            break
    return cabin, road, user_text


def _sent_context(trial: TrialResult) -> dict[str, Any] | None:
    if not trial.requests:
        return None
    marker = "CURRENT RELEVANT VEHICLE CONTEXT:\n"
    for message in trial.requests[0].get("messages", []):
        content = message.get("content", "")
        if not isinstance(content, str) or marker not in content:
            continue
        suffix = content.split(marker, 1)[1]
        try:
            value, _ = json.JSONDecoder().raw_decode(suffix.lstrip())
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def _sent_context_summary(selected: dict[str, Any] | None) -> str:
    if selected is None:
        return "<p class='empty'>本次首轮请求没有可解析的上下文；不要推断模型看到了上述观测。</p>"
    parts: list[str] = []
    driver = selected.get("driver", {})
    if isinstance(driver, dict):
        if driver.get("quality_status") == "KNOWN":
            facts = []
            if driver.get("state") is not None:
                facts.append(
                    f"驾驶员{_state(driver['state'], {'DROWSY': '疲劳', 'NORMAL': '正常', 'SUSPECTED': '疑似疲劳'})}"
                )
            if driver.get("eye_closure_seconds") is not None:
                facts.append(f"持续闭眼 {driver['eye_closure_seconds']} 秒")
            if driver.get("recent_yawns") is not None:
                facts.append(f"近期哈欠 {driver['recent_yawns']} 次")
            if driver.get("risk") is not None:
                facts.append(
                    f"风险{_state(driver['risk'], {'HIGH': '高', 'MEDIUM': '中', 'LOW': '低'})}"
                )
            if facts:
                parts.append("模型收到：" + "，".join(facts))
        else:
            parts.append("舱内状态不可作为当前可靠事实")
    road = selected.get("road", {})
    if isinstance(road, dict):
        if road.get("quality_status") == "KNOWN":
            facts = []
            if road.get("vehicle_count") is not None:
                facts.append(f"道路车辆 {road['vehicle_count']} 辆")
            if road.get("traffic_level") is not None:
                facts.append(
                    f"交通密度{_state(road['traffic_level'], {'MODERATE': '中等', 'LIGHT': '较低', 'HEAVY': '较高'})}"
                )
            for key, label in (
                ("lane_detected", "车道"),
                ("drivable_area_detected", "可行驶区"),
            ):
                if road.get(key) is not None:
                    facts.append(f"{label}{'已检测到' if road[key] else '未检测到'}")
            if facts:
                parts.append("模型收到：" + "，".join(facts))
        else:
            parts.append("舱外状态不可作为当前可靠事实")
    if not parts:
        return "<p class='empty'>本轮未选入舱内外关键信息；见折叠区核对原始请求。</p>"
    return "".join(f"<p class='input-summary'>{_escape(part)}</p>" for part in parts)


def _field(label: str, value: object, unit: str = "") -> str:
    text = "未提供" if value is None else f"{value}{unit}"
    return f"<div class='fact'><span>{_escape(label)}</span><strong>{_escape(text)}</strong></div>"


def _state(value: object, mapping: dict[str, str]) -> str:
    if value is None:
        return "未提供"
    return mapping.get(str(value), str(value))


def _observation_cards(cabin: dict, road: dict) -> str:
    if cabin:
        driver = "".join(
            (
                _field("持续闭眼", cabin.get("eye_closure_seconds"), " 秒"),
                _field("近期哈欠", cabin.get("recent_yawns"), " 次"),
                _field(
                    "驾驶状态",
                    _state(
                        cabin.get("driver_state"),
                        {
                            "DROWSY": "疲劳",
                            "NORMAL": "正常",
                            "SUSPECTED": "疑似疲劳",
                            "DISTRACTED": "分心",
                        },
                    ),
                ),
                _field(
                    "风险",
                    _state(
                        cabin.get("risk"),
                        {"HIGH": "高", "MEDIUM": "中", "LOW": "低", "CRITICAL": "严重"},
                    ),
                ),
            )
        )
    else:
        driver = "<p class='empty'>未提供舱内观测</p>"
    if road:
        outside = "".join(
            (
                _field("道路车辆", road.get("vehicle_count"), " 辆"),
                _field(
                    "行人 / 骑行者",
                    f"{road.get('pedestrian_count', '未知')} / {road.get('rider_count', '未知')}",
                ),
                _field(
                    "交通密度",
                    _state(
                        road.get("traffic_level"),
                        {"LIGHT": "较低", "MODERATE": "中等", "HEAVY": "较高"},
                    ),
                ),
                _field(
                    "车道",
                    _state(
                        road.get("lane_detected"),
                        {"True": "检测到", "False": "未检测到"},
                    ),
                ),
                _field(
                    "可行驶区",
                    _state(
                        road.get("drivable_area_detected"),
                        {"True": "检测到", "False": "未检测到"},
                    ),
                ),
            )
        )
    else:
        outside = "<p class='empty'>未提供舱外观测</p>"
    return (
        f"<article class='card'><div class='step'>01 · 录制观测</div><h2>舱内驾驶员</h2>{driver}</article>"
        f"<article class='card'><div class='step'>02 · 录制观测</div><h2>舱外道路</h2>{outside}</article>"
    )


def _decision_rows(trial: TrialResult) -> str:
    rows = []
    for index, response in enumerate(trial.model_responses, 1):
        content = response.get("content") or "本轮没有文本回复"
        rows.append(
            f"<div class='turn'><span>模型第 {index} 轮</span><p>{_escape(content)}</p></div>"
        )
        for call in response.get("tool_calls", []):
            rows.append(
                "<div class='turn tool'><span>模型请求工具</span>"
                f"<p>{_escape(call.get('name', '未知工具'))}"
                f" · {_escape(_json(call.get('arguments', {})))}</p></div>"
            )
    if not rows:
        return "<p class='empty'>模型没有返回可记录的回复</p>"
    return "".join(rows)


def _action_rows(trial: TrialResult) -> str:
    def tool_row(call: dict) -> str:
        return (
            "<div class='turn'><span>执行层记录</span>"
            f"<p>{_escape(call.get('name', '未知工具'))} · "
            f"{'成功' if call.get('success') else '未成功'} · "
            f"{'已确认' if call.get('confirmed') else '未确认或无需确认'}"
            f"{(' · ' + _escape(call['error'])) if call.get('error') else ''}</p></div>"
        )

    before_confirmation = "".join(
        tool_row(call) for call in trial.tool_calls if not call.get("confirmed")
    )
    after_confirmation = "".join(
        tool_row(call) for call in trial.tool_calls if call.get("confirmed")
    )
    if not trial.tool_calls:
        before_confirmation = "<p class='empty'>未请求工具；没有模拟车机动作</p>"
    interactions = "".join(
        f"<div class='turn'><span>{_escape(event.get('kind', '事件'))}</span>"
        f"<p>{_escape(event.get('text') or event.get('error') or ('成功' if event.get('success') else '未执行'))}</p></div>"
        for event in trial.interaction_events
        if event.get("kind") in {"confirmation", "rejection"}
    )
    navigation = trial.final_context.get("vehicle", {}).get("navigation_state")
    return (
        before_confirmation
        + interactions
        + after_confirmation
        + _field("最终模拟导航状态", navigation)
    )


def render_showcase(case: EvaluationCase, trial: TrialResult, grade: dict) -> str:
    """Render an offline-openable HTML report from one actual online trial."""
    cabin, road, user_text = _first_turn_observations(case)
    selected = _sent_context(trial)
    context_display = _sent_context_summary(selected)
    raw = {
        "首轮实际请求": trial.requests[0] if trial.requests else None,
        "模型响应": trial.model_responses,
        "工具执行": trial.tool_calls,
        "评分": grade,
    }
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VehicleMind · 在线 Agent 决策展示</title>
<style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#17212b;background:#f2f5f8}}
body{{max-width:1120px;margin:auto;padding:32px 20px 72px;line-height:1.6}}
h1{{font-size:clamp(30px,4vw,46px);margin:8px 0}}h2{{margin:4px 0 18px;font-size:21px}}
.eyebrow,.step{{color:#256f83;font-weight:750;letter-spacing:.04em}}.sub,.note{{color:#566473}}
.banner{{background:#fff4dc;border:1px solid #e8cd89;border-radius:14px;padding:14px 18px;margin:24px 0}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.card{{background:white;border:1px solid #dce4e9;border-radius:18px;padding:22px;margin:0 0 16px;box-shadow:0 6px 24px #18354a0a}}
.fact{{display:flex;justify-content:space-between;gap:18px;padding:9px 0;border-bottom:1px solid #edf1f4}}
.fact span{{color:#61717e}}.fact strong{{text-align:right}}.turn{{border-left:3px solid #69b4bc;padding:2px 0 2px 14px;margin:14px 0}}
.turn span{{font-size:13px;color:#526977;font-weight:700}}.turn p{{margin:4px 0;white-space:pre-wrap}}
.tool{{border-color:#e5a749}}.empty{{color:#647584}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fa;padding:14px;border-radius:10px;font-size:13px}}
.input-summary{{background:#edf6f7;border-left:4px solid #4b9bab;padding:10px 14px;border-radius:6px}}
details{{margin-top:14px}}summary{{cursor:pointer;font-weight:700}}.flow{{color:#235266;font-weight:700;margin:20px 0}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}body{{padding:18px 12px}}}}
</style></head><body>
<p class="eyebrow">VEHICLEMIND · 单次在线 Agent 决策</p><h1>从感知事实到车机建议</h1>
<p class="sub">场景 {_escape(case.id)} · {_escape(trial.provider)} / {_escape(trial.model)} · 第 {_escape(trial.trial_index)} 次运行</p>
<div class="banner">本页舱内外输入来自<strong>录制观测，非实时视频推理</strong>；模型回复来自本次真实在线调用。车辆状态和导航均为本地模拟。评分 {_escape(grade.get("status", "未评估"))} 不是独立人工判定。</div>
<div class="grid">{_observation_cards(cabin, road)}</div>
<div class="flow">录制观测 → 上下文筛选 → 在线模型 → 工具确认与模拟结果</div>
<section class="card"><div class="step">03 · 用户请求与模型输入</div><h2>Agent 实际看到了什么？</h2>
<p><strong>用户：</strong>{_escape(user_text)}</p><p class="note">下方来自首轮实际发送给模型的消息；上方观测不一定全部入选。</p>
<h3>实际发送给模型的上下文</h3>{context_display}</section>
<div class="grid"><section class="card"><div class="step">04 · 在线模型输出</div><h2>模型给出什么建议？</h2>{_decision_rows(trial)}</section>
<section class="card"><div class="step">05 · 执行层</div><h2>实际发生了什么？</h2>{_action_rows(trial)}</section></div>
<section class="card"><div class="step">证据与边界</div><h2>逐项核对</h2>
<p>模型请求 {_escape(trial.request_count)} 次 · 耗时 {_escape(round(trial.latency_ms, 1))} 毫秒 · 错误 {_escape(trial.error or "无")}。</p>
<p class="note">高风险疲劳时，播放音乐不能等同于停车休息；本页只陈述模型实际输出，不自动认可建议安全性。</p>
<details><summary>展开原始请求、响应、工具与评分</summary><pre>{_escape(_json(raw))}</pre></details>
<p class="note">完整证据见同目录 trial.json；候选场景和 AI 自审场景均不是独立人工金标。</p></section>
</body></html>"""

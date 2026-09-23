"""Chinese presentation of demo diagnostics; core protocol values stay unchanged."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from apps.vehicle_ai_demo.quality_display import format_age
from modules.vehicle_ai.events.events import VehicleEvent


LABELS = {
    "driver": "驾驶员",
    "road": "道路",
    "vehicle": "车辆",
    "presence": "是否在位",
    "state": "驾驶状态",
    "risk": "风险等级",
    "perclos": "闭眼比例",
    "eye_closed": "闭眼",
    "eye_closure_seconds": "持续闭眼秒数",
    "recent_yawns": "近期哈欠次数",
    "vehicle_count": "车辆数",
    "pedestrian_count": "行人数",
    "rider_count": "骑行者数",
    "traffic_light_count": "交通灯数",
    "traffic_sign_count": "交通标志数",
    "total_objects": "目标总数",
    "lane_detected": "检测到车道",
    "drivable_area_detected": "检测到可行驶区域",
    "traffic_level": "交通密度",
    "speed_kmh": "车速",
    "gear": "挡位",
    "cabin_temperature_c": "舱内温度",
    "target_temperature_c": "目标温度",
    "ac_enabled": "空调开启",
    "driver_window_open": "驾驶员车窗开启",
    "media_playing": "媒体播放中",
    "media_title": "媒体标题",
    "volume": "音量",
    "navigation_state": "导航状态",
    "navigation_destination_id": "目的地标识",
    "navigation_destination": "导航目的地",
    "old_risk": "原风险",
    "new_risk": "新风险",
    "old_state": "原状态",
    "new_state": "新状态",
    "old_presence": "原在位状态",
    "new_presence": "新在位状态",
    "old_level": "原交通密度",
    "new_level": "新交通密度",
    "driver_state": "驾驶状态",
    "destination": "目的地",
}
VALUES = {
    "PRESENT": "在位",
    "ABSENT": "不在位",
    "UNKNOWN": "未知",
    "NORMAL": "正常",
    "DROWSY": "疲劳",
    "DISTRACTED": "分心",
    "LOW": "低",
    "MEDIUM": "中",
    "HIGH": "高",
    "CRITICAL": "严重",
    "MODERATE": "中等",
    "IDLE": "未启动",
    "ACTIVE": "进行中",
    "MISSING": "缺失",
    "INVALID": "无效",
    "STALE": "已过期",
    "KNOWN": "有效",
    "D": "前进挡（D）",
    "P": "驻车挡（P）",
    "R": "倒车挡（R）",
    "N": "空挡（N）",
    "cabin_perception": "舱内感知",
    "driving_perception": "道路感知",
}
EVENTS = {
    "DRIVER_STATE_CHANGED": "驾驶状态变化",
    "DRIVER_RISK_CHANGED": "驾驶风险变化",
    "DRIVER_ABSENT": "驾驶员不在位",
    "DRIVER_PRESENT": "驾驶员在位",
    "HIGH_RISK_DETECTED": "检测到高风险",
    "TRAFFIC_LEVEL_CHANGED": "交通密度变化",
    "LANE_LOST": "车道丢失",
    "DRIVABLE_AREA_LOST": "可行驶区域丢失",
    "NAVIGATION_STATE_CHANGED": "导航状态变化",
    "USER_UTTERANCE": "用户输入",
}
STAGES = {"capture": "采集", "infer": "推理", "update": "上下文更新", "display": "展示"}
QUEUES = {"frames": "视频帧", "snapshots": "感知快照", "display": "展示"}


def _value(value: Any) -> str:
    if value is None:
        return "未提供"
    if isinstance(value, bool):
        return "是" if value else "否"
    return VALUES.get(str(value), str(value))


def format_context(context: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["统一上下文："]
    for domain, fields in context.items():
        lines.append(f"{LABELS.get(domain, domain)}：")
        for key, value in fields.items():
            lines.append(f"  {LABELS.get(key, key)}：{_value(value)}")
    return "\n".join(lines)


def format_freshness(freshness: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["感知新鲜度："]
    for domain, info in freshness.items():
        lines.append(
            f"  {LABELS.get(domain, domain)}：距上次更新 {format_age(info['age_seconds'])}；"
            f"状态 {_value(info['status'])}"
        )
    return "\n".join(lines)


def format_event(event: VehicleEvent) -> str:
    label = EVENTS.get(str(event.type), f"未翻译事件：{event.type}")
    priority = _value(event.priority)
    detail = "；".join(
        f"{LABELS.get(key, key)}：{_value(value)}" for key, value in event.data.items()
    )
    suffix = f"；{detail}" if detail else ""
    return f"[{priority}] {label}；来源：{_value(event.source)}{suffix}"


def format_health(health: Mapping[str, Any]) -> str:
    name = {"cabin": "舱内", "road": "道路"}.get(
        str(health.get("name")), str(health.get("name", "未知"))
    )
    lines = [f"{name}流水线："]
    error = health.get("last_error")
    lines.append(f"  最近错误：{error if error else '无'}")
    latency = health.get("event_publish_p95_ms")
    lines.append(
        f"  事件发布 p95 延迟：{latency if latency is not None else '未观测'} 毫秒"
    )
    for stage, stats in health.get("stages", {}).items():
        lines.append(
            f"  {STAGES.get(stage, stage)}阶段：处理数 {stats['processed']}；"
            f"p95 延迟 {stats['p95_latency_ms'] if stats['p95_latency_ms'] is not None else '未观测'} 毫秒"
        )
    for queue, stats in health.get("queues", {}).items():
        lines.append(
            f"  {QUEUES.get(queue, queue)}队列：深度 {stats['depth']}；丢弃数 {stats['dropped']}"
        )
    return "\n".join(lines)

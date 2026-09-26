"""Build a small, quality-aware list of facts a response must cover."""

from __future__ import annotations

from typing import Any

from modules.vehicle_ai.context.context_manager import ContextManager


_DRIVER_FIELDS = ("presence", "state", "risk")
_DRIVER_ASSESSMENT_FIELDS = ("state", "risk")
_HIGH_RISKS = {"HIGH", "CRITICAL"}
_STATUS_ZH = {
    "MISSING": "缺失",
    "INVALID": "无效",
    "STALE": "已过期",
    "UNKNOWN": "未知",
}
_DOMAIN_ZH = {"driver": "驾驶员", "road": "道路"}


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def build_decision_brief(
    context: dict[str, Any], manager: ContextManager
) -> dict[str, Any]:
    """Describe only selected values and unavailable quality states."""
    required_points: list[dict[str, Any]] = []
    unavailable_fields: dict[str, str] = {}

    def add_point(code: str, text: str, fields: list[str]) -> None:
        required_points.append(
            {"code": code, "text_zh": text, "evidence_fields": fields}
        )

    driver = context.get("driver")
    if isinstance(driver, dict):
        field_statuses = {
            field: _value(manager.field_quality("driver", field))
            for field in _DRIVER_FIELDS
        }
        for field, status in field_statuses.items():
            if status != "KNOWN":
                unavailable_fields[f"driver.{field}"] = status

        if (
            field_statuses["presence"] == "KNOWN"
            and "presence" in driver
            and _value(driver["presence"]) == "ABSENT"
        ):
            add_point(
                "DRIVER_NOT_DETECTED",
                "观测未检测到驾驶员，不能据此断言车内无人。",
                ["driver.presence"],
            )

        unavailable_assessment = [
            f"driver.{field}"
            for field in _DRIVER_ASSESSMENT_FIELDS
            if field_statuses[field] != "KNOWN"
        ]
        if unavailable_assessment:
            add_point(
                "DRIVER_STATE_UNAVAILABLE",
                "驾驶状态或风险信息不足，分别按字段说明，不推断正常或疲劳。",
                unavailable_assessment,
            )

        if (
            field_statuses["risk"] == "KNOWN"
            and "risk" in driver
            and _value(driver["risk"]) in _HIGH_RISKS
        ):
            evidence_fields = ["driver.risk"]
            if field_statuses["state"] == "KNOWN" and "state" in driver:
                evidence_fields.append("driver.state")
            add_point(
                "DRIVER_RISK_HIGH",
                "当前有效记录标示驾驶风险较高；仅按已提供字段说明。",
                evidence_fields,
            )

    for domain in ("driver", "road"):
        values = context.get(domain)
        if not isinstance(values, dict):
            continue
        status = _value(values.get("quality_status", "UNKNOWN"))
        if status == "KNOWN":
            continue
        domain_name = _DOMAIN_ZH[domain]
        status_name = _STATUS_ZH.get(status, status)
        add_point(
            "DOMAIN_UNAVAILABLE",
            f"{domain_name}观测{status_name}，不能据此判断本轮当前状态。",
            [f"{domain}.quality_status"],
        )

    return {
        "version": 1,
        "required_points": required_points,
        "unavailable_fields": unavailable_fields,
    }

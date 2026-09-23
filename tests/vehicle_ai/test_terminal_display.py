from apps.vehicle_ai_demo.terminal_display import (
    format_context,
    format_event,
    format_freshness,
    format_health,
)
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.events.events import (
    EventPriority,
    EventType,
    VehicleEvent,
)


def test_context_and_freshness_are_chinese() -> None:
    manager = ContextManager()
    text = format_context(manager.get_agent_context())
    assert "驾驶员" in text
    assert "道路" in text
    assert "未知" in text
    assert "presence" not in text

    freshness = format_freshness(
        {"driver": {"age_seconds": None, "fresh": False, "status": "MISSING"}}
    )
    assert "未观测" in freshness
    assert "缺失" in freshness


def test_event_and_health_are_chinese() -> None:
    event = VehicleEvent(
        type=EventType.HIGH_RISK_DETECTED,
        priority=EventPriority.CRITICAL,
        source="cabin_perception",
        message="High risk detected",
        data={"old_risk": "LOW", "new_risk": "HIGH", "custom_code": "X1"},
    )
    assert "检测到高风险" in format_event(event)
    assert "严重" in format_event(event)
    assert "High risk detected" not in format_event(event)
    assert "原风险：低" in format_event(event)
    assert "新风险：高" in format_event(event)
    assert "custom_code：X1" in format_event(event)

    health = format_health(
        {
            "name": "cabin",
            "last_error": None,
            "event_publish_p95_ms": 12.5,
            "stages": {
                "infer": {"processed": 3, "heartbeat_at": None, "p95_latency_ms": 4.0}
            },
            "queues": {"frames": {"depth": 1, "dropped": 2}},
        }
    )
    assert "舱内流水线" in health
    assert "事件发布 p95 延迟" in health
    assert "丢弃数" in health
    assert "last_error" not in health

import json

import pytest

from modules.observation import ObservationMetadata
from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent
from modules.vehicle_ai.context import ContextManager, RiskLevel
from modules.vehicle_ai.context.context_selector import ContextSelector


def context_message(manager, text):
    agent = VehicleAgent.__new__(VehicleAgent)
    agent.context_manager = manager
    agent.context_selector = ContextSelector()
    return agent._context_message(text)["content"]


@pytest.mark.parametrize("text", ["放点音乐", "导航到附近停车场", "我想休息"])
def test_action_requests_include_driver_and_road_risk(text):
    manager = ContextManager()
    manager.update_driver(risk=RiskLevel.HIGH)
    manager.update_road(vehicle_count=3)
    payload = context_message(manager, text)
    assert '"risk": "HIGH"' in payload
    assert '"vehicle_count": 3' in payload


def test_field_evidence_keeps_original_source_after_partial_update():
    manager = ContextManager()
    metadata = ObservationMetadata(120, 2, "cabin_video", None, True, 3.0)
    manager.update_driver(observation=metadata, risk=RiskLevel.HIGH)
    manager.update_driver(recent_yawns=2)
    payload = context_message(manager, "驾驶状态")
    evidence = json.JSONDecoder().raw_decode(payload.split("CONTEXT:\n", 1)[1])[0]
    risk = evidence["driver"]["field_evidence"]["risk"]
    assert risk["source"] == "cabin_video"
    assert risk["timestamp_ms"] == 120
    assert risk["confidence"] is None
    assert evidence["driver"]["field_evidence"]["recent_yawns"]["source"] is None


def test_unobserved_self_report_does_not_become_sensor_evidence():
    payload = context_message(ContextManager(), "我很困，放音乐")
    assert '"risk": "HIGH"' not in payload
    assert '"quality_status": "MISSING"' in payload
    assert "User statements are not sensor observations" in payload


def test_generic_chat_keeps_context_empty():
    assert "No vehicle context" in context_message(ContextManager(), "你好")


@pytest.mark.parametrize("invalid", [False, True])
def test_unusable_observation_does_not_leak_value_or_field_evidence(invalid):
    clock = [10.0]
    manager = ContextManager(quality_clock=lambda: clock[0])
    manager.update_driver(risk=RiskLevel.HIGH)
    if invalid:
        manager.mark_invalid_observation(
            "driver", ObservationMetadata(10, 0, "cabin", None, False, 0.0)
        )
    else:
        clock[0] += 3
    payload = context_message(manager, "放音乐")
    assert '"risk": "HIGH"' not in payload
    expected = "INVALID" if invalid else "STALE"
    assert f'"quality_status": "{expected}"' in payload


def test_navigation_aliases_use_vehicle_field_evidence():
    manager = ContextManager()
    manager.update_vehicle(navigation_destination="服务区")
    payload = context_message(manager, "导航")
    data = json.JSONDecoder().raw_decode(payload.split("CONTEXT:\n", 1)[1])[0]
    assert (
        data["navigation"]["field_evidence"]["destination"]["quality_status"] == "KNOWN"
    )

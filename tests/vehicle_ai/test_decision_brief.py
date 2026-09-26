import json

import pytest

from modules.observation import ObservationMetadata
from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent
from modules.vehicle_ai.context import ContextManager, DriverState, RiskLevel
from modules.vehicle_ai.context.context_selector import ContextSelector
from modules.vehicle_ai.context.enums import DriverPresence


def context_message(manager, text):
    agent = VehicleAgent.__new__(VehicleAgent)
    agent.context_manager = manager
    agent.context_selector = ContextSelector()
    return agent._context_message(text)["content"]


def recorded_brief(payload):
    marker = "DECISION BRIEF:\n"
    assert marker in payload, "the actual request should record its decision brief"
    return json.JSONDecoder().raw_decode(payload.split(marker, 1)[1])[0]


def recorded_context(payload):
    return json.JSONDecoder().raw_decode(payload.split("CONTEXT:\n", 1)[1])[0]


def test_absence_and_unknown_driver_fields_are_explained_in_request_brief():
    manager = ContextManager()
    manager.update_driver(
        presence=DriverPresence.ABSENT,
        state=DriverState.UNKNOWN,
        risk=RiskLevel.UNKNOWN,
    )

    payload = context_message(manager, "当前舱内有驾驶员吗？能判断驾驶状态吗？")
    brief = recorded_brief(payload)
    points = {point["code"]: point for point in brief["required_points"]}

    assert points["DRIVER_NOT_DETECTED"]["evidence_fields"] == ["driver.presence"]
    assert "不能据此断言车内无人" in points["DRIVER_NOT_DETECTED"]["text_zh"]
    assert "DRIVER_STATE_UNAVAILABLE" in points
    assert brief["unavailable_fields"]["driver.state"] == "UNKNOWN"
    assert brief["unavailable_fields"]["driver.risk"] == "UNKNOWN"


def test_fresh_driver_point_survives_stale_road_in_brief():
    clock = [0.0]
    manager = ContextManager(quality_clock=lambda: clock[0])
    manager.update_road(vehicle_count=11)
    clock[0] = 1.5
    manager.update_driver(state=DriverState.DROWSY, risk=RiskLevel.HIGH)
    clock[0] = 1.6

    payload = context_message(manager, "结合我现在的状态和道路情况说说。")
    brief = recorded_brief(payload)
    codes = {point["code"] for point in brief["required_points"]}
    selected = recorded_context(payload)

    assert "DRIVER_RISK_HIGH" in codes
    assert "DOMAIN_UNAVAILABLE" in codes
    assert selected["road"]["quality_status"] == "STALE"
    assert selected["driver"]["risk"] == "HIGH"


def test_stale_driver_value_is_not_revived_by_fresh_road():
    clock = [0.0]
    manager = ContextManager(quality_clock=lambda: clock[0])
    manager.update_driver(risk=RiskLevel.HIGH)
    clock[0] = 2.1
    manager.update_road(vehicle_count=11)
    clock[0] = 2.2

    payload = context_message(manager, "驾驶员风险和道路情况如何？")
    brief = recorded_brief(payload)
    codes = {point["code"] for point in brief["required_points"]}
    selected = recorded_context(payload)

    assert "DRIVER_RISK_HIGH" not in codes
    assert selected["driver"]["quality_status"] == "STALE"
    assert selected["road"]["vehicle_count"] == 11


def test_fresh_driver_field_does_not_revive_stale_risk_field():
    clock = [0.0]
    manager = ContextManager(quality_clock=lambda: clock[0])
    manager.update_driver(risk=RiskLevel.HIGH)
    clock[0] = 2.1
    manager.update_driver(state=DriverState.NORMAL)

    payload = context_message(manager, "驾驶状态和风险如何？")
    brief = recorded_brief(payload)
    points = {point["code"] for point in brief["required_points"]}
    selected = recorded_context(payload)

    assert "state" in selected["driver"]
    assert "risk" not in selected["driver"]
    assert "DRIVER_RISK_HIGH" not in points
    assert brief["unavailable_fields"]["driver.risk"] == "STALE"
    assert "DRIVER_STATE_UNAVAILABLE" in points


@pytest.mark.parametrize("invalid", [False, True])
def test_missing_or_invalid_driver_context_has_no_unsupported_risk(invalid):
    manager = ContextManager()
    if invalid:
        manager.update_driver(risk=RiskLevel.HIGH)
        manager.mark_invalid_observation(
            "driver", ObservationMetadata(10, 0, "cabin", None, False, 0.0)
        )

    payload = context_message(manager, "请说明当前驾驶状态。")
    brief = recorded_brief(payload)
    selected = recorded_context(payload)
    codes = {point["code"] for point in brief["required_points"]}
    expected = "INVALID" if invalid else "MISSING"

    assert "risk" not in selected["driver"]
    assert "DRIVER_RISK_HIGH" not in codes
    assert brief["unavailable_fields"]["driver.risk"] == expected


def test_generic_greeting_does_not_get_a_decision_brief():
    payload = context_message(ContextManager(), "你好")

    assert "No vehicle context" in payload
    assert "DECISION BRIEF:" not in payload

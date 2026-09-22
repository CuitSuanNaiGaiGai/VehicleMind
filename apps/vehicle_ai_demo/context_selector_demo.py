import json

from modules.vehicle_ai.context import (
    ContextManager,
    ContextSelector,
    DriverPresence,
    DriverState,
    GearState,
    RiskLevel,
)


def test(
    selector,
    context,
    text,
):
    print()
    print("========================================")

    print(f"User: {text}")

    result = selector.select(
        user_text=text,
        vehicle_context=context,
    )

    print(
        "Topics:",
        [topic.value for topic in result.topics],
    )

    print(
        "Matched:",
        result.matched_keywords,
    )

    print("Selected Context:")

    print(
        json.dumps(
            result.context,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def main():

    manager = ContextManager()

    manager.update_vehicle(
        speed_kmh=68.0,
        gear=GearState.D,
        cabin_temperature_c=28.0,
        target_temperature_c=24.0,
        ac_enabled=True,
        volume=25,
    )

    manager.update_driver(
        presence=(DriverPresence.PRESENT),
        state=(DriverState.DROWSY),
        risk=(RiskLevel.HIGH),
        perclos=0.31,
        eye_closed=True,
        eye_closure_seconds=2.2,
        recent_yawns=2,
    )

    manager.update_road(
        vehicle_count=14,
        pedestrian_count=2,
        rider_count=1,
        traffic_light_count=2,
        traffic_sign_count=3,
        lane_detected=True,
        drivable_area_detected=True,
        traffic_level="HEAVY",
    )

    context = manager.get_context()

    selector = ContextSelector()

    test(
        selector,
        context,
        "你好",
    )

    test(
        selector,
        context,
        "车里有点热",
    )

    test(
        selector,
        context,
        "我有点困",
    )

    test(
        selector,
        context,
        "我有点困，帮我找个地方休息",
    )

    test(
        selector,
        context,
        "前面车多吗？",
    )

    test(
        selector,
        context,
        "播放一点轻音乐",
    )

    test(
        selector,
        context,
        "我们现在开多快？",
    )


if __name__ == "__main__":
    main()

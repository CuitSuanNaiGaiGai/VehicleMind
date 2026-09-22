import time

from apps.vehicle_ai_demo.quality_display import format_age
from modules.vehicle_ai.context import (
    ContextManager,
    DriverPresence,
    DriverState,
    GearState,
    RiskLevel,
)


def print_changes(
    title,
    changes,
):
    print()
    print(f"========== {title} ==========")

    if not changes:
        print("No semantic change.")

        return

    for change in changes:
        print(
            "[CHANGE]",
            change,
        )


def main():
    # ========================================================
    # Context Manager
    # ========================================================

    manager = ContextManager()

    # ========================================================
    # Initial context
    # ========================================================

    print(manager.summary())

    # ========================================================
    # Vehicle startup
    # ========================================================

    changes = manager.update_vehicle(
        gear=GearState.D,
        speed_kmh=45.0,
        cabin_temperature_c=27.5,
        target_temperature_c=24.0,
        ac_enabled=True,
    )

    print_changes(
        "VEHICLE UPDATE",
        changes,
    )

    # ========================================================
    # Cabin perception
    # ========================================================

    changes = manager.update_driver(
        presence=(DriverPresence.PRESENT),
        state=(DriverState.NORMAL),
        risk=(RiskLevel.LOW),
        perclos=0.12,
        eye_closed=False,
        eye_closure_seconds=0.0,
        recent_yawns=0,
    )

    print_changes(
        "DRIVER UPDATE",
        changes,
    )

    # ========================================================
    # Driving perception
    # ========================================================

    changes = manager.update_road(
        vehicle_count=5,
        pedestrian_count=1,
        rider_count=0,
        traffic_light_count=2,
        traffic_sign_count=1,
        total_objects=9,
        lane_detected=True,
        drivable_area_detected=True,
        traffic_level="MODERATE",
    )

    print_changes(
        "ROAD UPDATE",
        changes,
    )

    # ========================================================
    # No-change test
    #
    # Same value should NOT produce an event.
    # ========================================================

    changes = manager.update_vehicle(speed_kmh=45.0)

    print_changes(
        "IDENTICAL VALUE",
        changes,
    )

    # ========================================================
    # Simulate driving
    # ========================================================

    time.sleep(0.2)

    changes = manager.update_vehicle(speed_kmh=68.0)

    print_changes(
        "SPEED CHANGE",
        changes,
    )

    # ========================================================
    # Simulate fatigue transition
    # ========================================================

    time.sleep(0.2)

    changes = manager.update_driver(
        state=(DriverState.SUSPECTED),
        risk=(RiskLevel.MEDIUM),
        perclos=0.27,
        recent_yawns=2,
    )

    print_changes(
        "FATIGUE WARNING",
        changes,
    )

    # ========================================================
    # Simulate DROWSY
    # ========================================================

    time.sleep(0.2)

    changes = manager.update_driver(
        state=(DriverState.DROWSY),
        risk=(RiskLevel.HIGH),
        eye_closed=True,
        eye_closure_seconds=2.1,
    )

    print_changes(
        "DROWSY EVENT",
        changes,
    )

    # ========================================================
    # Current snapshot
    # ========================================================

    print()
    print(manager.summary())

    # ========================================================
    # Agent-facing context
    # ========================================================

    print("========== AGENT CONTEXT ==========")

    context = manager.get_context()

    print(context.to_agent_json())

    # ========================================================
    # Freshness
    # ========================================================

    print()
    print("========== FRESHNESS ==========")

    freshness = manager.freshness()

    for domain, info in freshness.items():
        print(
            f"{domain:<8} age={format_age(info['age_seconds'])} fresh={info['fresh']}"
        )

    # ========================================================
    # Recent changes
    # ========================================================

    print()
    print("========== CHANGE HISTORY ==========")

    for change in manager.recent_changes(limit=20):
        print(change)


if __name__ == "__main__":
    main()

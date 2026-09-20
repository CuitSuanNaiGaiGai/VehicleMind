from modules.vehicle_ai.context import (
    ContextManager,
    DriverPresence,
    DriverState,
    GearState,
    NavigationState,
    RiskLevel,
)

from modules.vehicle_ai.events import (
    EventBus,
    EventDetector,
    EventPriority,
    EventType,
    VehicleEvent,
    create_user_utterance_event,
)


# ============================================================
# Event logger
# ============================================================


def event_logger(
    event: VehicleEvent,
) -> None:

    print(
        f"[EVENT] "
        f"[{event.priority}] "
        f"{event.type}"
    )

    print(
        f"        "
        f"{event.message}"
    )

    if event.data:

        print(
            f"        "
            f"{event.data}"
        )


# ============================================================
# Critical-event handler
# ============================================================


def critical_handler(
    event: VehicleEvent,
) -> None:

    if (
        event.priority
        == EventPriority.CRITICAL
    ):

        print()
        print(
            ">>> SAFETY-RELEVANT EVENT <<<"
        )

        print(
            event.message
        )

        print()


# ============================================================
# Process changes
# ============================================================


def process_changes(
    manager: ContextManager,
    detector: EventDetector,
    bus: EventBus,
    changes,
) -> None:

    context = (
        manager.get_context()
    )

    events = (
        detector.detect(
            changes=changes,
            context=context,
        )
    )

    bus.publish_many(
        events
    )


# ============================================================
# Main
# ============================================================


def main():
    # ========================================================
    # Infrastructure
    # ========================================================

    manager = (
        ContextManager()
    )

    detector = (
        EventDetector()
    )

    bus = (
        EventBus()
    )

    # --------------------------------------------------------
    # Logger receives every event.
    # --------------------------------------------------------

    bus.subscribe_all(
        event_logger
    )

    # --------------------------------------------------------
    # Example of a dedicated subscriber.
    # --------------------------------------------------------

    bus.subscribe(
        EventType.HIGH_RISK_DETECTED,
        critical_handler,
    )

    print()
    print(
        "========================================"
    )

    print(
        " VehicleMind Event Demo"
    )

    print(
        "========================================"
    )

    # ========================================================
    # Vehicle startup
    # ========================================================

    changes = (
        manager.update_vehicle(
            gear=GearState.D,
            speed_kmh=45.0,
            cabin_temperature_c=27.0,
            ac_enabled=True,
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # --------------------------------------------------------
    # Notice:
    #
    # speed / temperature changes do NOT create semantic
    # events with the current EventDetector.
    # --------------------------------------------------------

    # ========================================================
    # Driver appears
    # ========================================================

    print()
    print(
        "--- Driver detected ---"
    )

    changes = (
        manager.update_driver(
            presence=(
                DriverPresence.PRESENT
            ),
            state=(
                DriverState.NORMAL
            ),
            risk=(
                RiskLevel.LOW
            ),
            perclos=0.10,
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # ========================================================
    # Normal speed changes
    #
    # These should generate no semantic events.
    # ========================================================

    print()
    print(
        "--- Normal speed updates ---"
    )

    for speed in (
        50.0,
        55.0,
        61.0,
        68.0,
    ):

        changes = (
            manager.update_vehicle(
                speed_kmh=speed
            )
        )

        process_changes(
            manager,
            detector,
            bus,
            changes,
        )

    print(
        "No agent-level speed events generated."
    )

    # ========================================================
    # Fatigue begins
    # ========================================================

    print()
    print(
        "--- Driver fatigue transition ---"
    )

    changes = (
        manager.update_driver(
            state=(
                DriverState.SUSPECTED
            ),
            risk=(
                RiskLevel.MEDIUM
            ),
            perclos=0.27,
            recent_yawns=2,
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # ========================================================
    # DROWSY / HIGH
    # ========================================================

    print()
    print(
        "--- High-risk transition ---"
    )

    changes = (
        manager.update_driver(
            state=(
                DriverState.DROWSY
            ),
            risk=(
                RiskLevel.HIGH
            ),
            eye_closed=True,
            eye_closure_seconds=2.2,
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # ========================================================
    # Road condition change
    # ========================================================

    print()
    print(
        "--- Road context ---"
    )

    # Initial stable road scene
    changes = (
        manager.update_road(
            traffic_level="MODERATE",
            lane_detected=True,
            drivable_area_detected=True,
            vehicle_count=6,
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # Traffic becomes heavy
    changes = (
        manager.update_road(
            traffic_level="HEAVY",
            vehicle_count=14,
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # Lane temporarily unavailable
    changes = (
        manager.update_road(
            lane_detected=False
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # ========================================================
    # Navigation
    # ========================================================

    print()
    print(
        "--- Navigation ---"
    )

    changes = (
        manager.update_vehicle(
            navigation_destination=(
                "Nearby Rest Area"
            ),
            navigation_state=(
                NavigationState.ACTIVE
            ),
        )
    )

    process_changes(
        manager,
        detector,
        bus,
        changes,
    )

    # ========================================================
    # User utterance
    # ========================================================

    print()
    print(
        "--- User Interaction ---"
    )

    user_event = (
        create_user_utterance_event(
            "我有点困，帮我找个地方休息。"
        )
    )

    bus.publish(
        user_event
    )

    # ========================================================
    # Event queue
    # ========================================================

    print()
    print(
        "========== PENDING EVENTS =========="
    )

    pending = (
        bus.pending_events()
    )

    for event in pending:

        print(
            event
        )

    # ========================================================
    # Consume
    # ========================================================

    consumed = (
        bus.consume_pending()
    )

    print()
    print(
        f"Consumed events: "
        f"{len(consumed)}"
    )

    print(
        f"Pending after consume: "
        f"{len(bus.pending_events())}"
    )


if __name__ == "__main__":
    main()

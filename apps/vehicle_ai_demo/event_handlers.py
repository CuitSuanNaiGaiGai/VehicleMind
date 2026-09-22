from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.events import (
    EventBus,
    EventDetector,
    EventPriority,
    VehicleEvent,
)


# ============================================================
# Event logger
# ============================================================


def event_logger(
    event: VehicleEvent,
) -> None:

    print(f"[EVENT] [{event.priority}] {event.type}")

    print(f"        {event.message}")

    if event.data:
        print(f"        {event.data}")


# ============================================================
# Critical-event handler
# ============================================================


def critical_handler(
    event: VehicleEvent,
) -> None:

    if event.priority == EventPriority.CRITICAL:
        print()
        print(">>> SAFETY-RELEVANT EVENT <<<")

        print(event.message)

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

    context = manager.get_context()

    events = detector.detect(
        changes=changes,
        context=context,
    )

    bus.publish_many(events)

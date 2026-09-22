from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.integration import (
    CabinContextAdapter,
)


def main():

    manager = ContextManager()

    adapter = CabinContextAdapter(
        manager
    )

    # ========================================================
    # Simulate one Cabin Intelligence frame
    # ========================================================

    changes = adapter.update(
        presence="PRESENT",
        driver_state="NORMAL",
        risk="LOW",
        perclos=0.12,
        eye_closed=False,
        eye_closure_seconds=0.0,
        recent_yawns=0,
        blink_count=12,
    )

    print(
        "========== NORMAL =========="
    )

    for change in changes:
        print(change)

    print(
        manager.summary()
    )

    # ========================================================
    # Simulate fatigue
    # ========================================================

    changes = adapter.update(
        presence="PRESENT",
        driver_state="DROWSY",
        risk="HIGH",
        perclos=0.32,
        eye_closed=True,
        eye_closure_seconds=2.3,
        recent_yawns=2,
        blink_count=18,
    )

    print(
        "========== DROWSY =========="
    )

    for change in changes:
        print(change)

    print(
        manager.summary()
    )


if __name__ == "__main__":
    main()

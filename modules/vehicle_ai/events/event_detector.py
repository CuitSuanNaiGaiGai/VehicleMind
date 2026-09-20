from __future__ import annotations

from typing import Any

from modules.vehicle_ai.context import (
    ContextChange,
    ContextDomain,
    DriverPresence,
    DriverState,
    RiskLevel,
    VehicleContext,
)

from modules.vehicle_ai.events.events import (
    EventPriority,
    EventType,
    VehicleEvent,
)


# ============================================================
# Helpers
# ============================================================


def normalize_value(
    value: Any,
) -> str:
    """
    Normalize enum/string values for comparison.
    """

    if hasattr(
        value,
        "value",
    ):

        return str(
            value.value
        )

    return str(
        value
    )


# ============================================================
# Event Detector
# ============================================================


class EventDetector:
    """
    Convert low-level ContextChange objects into semantic
    VehicleEvent objects.

    Example:

        ContextChange

            driver.state
            NORMAL -> DROWSY

        becomes

            DRIVER_STATE_CHANGED

    while:

        vehicle.speed_kmh
        67.8 -> 68.0

    produces no event at this layer.

    The purpose is to prevent noisy high-frequency vehicle
    context updates from directly triggering the future LLM.
    """

    def detect(
        self,
        changes: list[ContextChange],
        context: VehicleContext,
    ) -> list[VehicleEvent]:

        events: list[
            VehicleEvent
        ] = []

        for change in changes:

            if (
                change.domain
                == ContextDomain.DRIVER
            ):

                events.extend(
                    self._driver_events(
                        change,
                        context,
                    )
                )

            elif (
                change.domain
                == ContextDomain.ROAD
            ):

                events.extend(
                    self._road_events(
                        change,
                        context,
                    )
                )

            elif (
                change.domain
                == ContextDomain.VEHICLE
            ):

                events.extend(
                    self._vehicle_events(
                        change,
                        context,
                    )
                )

        return events

    # ========================================================
    # Driver events
    # ========================================================

    def _driver_events(
        self,
        change: ContextChange,
        context: VehicleContext,
    ) -> list[VehicleEvent]:

        events: list[
            VehicleEvent
        ] = []

        old_value = normalize_value(
            change.old_value
        )

        new_value = normalize_value(
            change.new_value
        )

        # ----------------------------------------------------
        # Driver state transition
        # ----------------------------------------------------

        if (
            change.field
            == "state"
        ):

            if (
                new_value
                == DriverState.DROWSY.value
            ):

                priority = (
                    EventPriority.HIGH
                )

            elif (
                new_value
                == DriverState.SUSPECTED.value
            ):

                priority = (
                    EventPriority.MEDIUM
                )

            else:

                priority = (
                    EventPriority.LOW
                )

            events.append(
                VehicleEvent(
                    type=(
                        EventType
                        .DRIVER_STATE_CHANGED
                    ),
                    priority=priority,
                    source="cabin_perception",
                    message=(
                        "Driver state changed "
                        f"from {old_value} "
                        f"to {new_value}."
                    ),
                    data={
                        "old_state":
                            old_value,

                        "new_state":
                            new_value,

                        "risk":
                            normalize_value(
                                context
                                .driver
                                .risk
                            ),

                        "perclos":
                            context
                            .driver
                            .perclos,

                        "eye_closure_seconds":
                            context
                            .driver
                            .eye_closure_seconds,

                        "recent_yawns":
                            context
                            .driver
                            .recent_yawns,
                    },
                )
            )

        # ----------------------------------------------------
        # Driver risk transition
        # ----------------------------------------------------

        elif (
            change.field
            == "risk"
        ):

            if (
                new_value
                == RiskLevel.HIGH.value
            ):

                priority = (
                    EventPriority.HIGH
                )

            elif (
                new_value
                == RiskLevel.MEDIUM.value
            ):

                priority = (
                    EventPriority.MEDIUM
                )

            else:

                priority = (
                    EventPriority.LOW
                )

            events.append(
                VehicleEvent(
                    type=(
                        EventType
                        .DRIVER_RISK_CHANGED
                    ),
                    priority=priority,
                    source="cabin_perception",
                    message=(
                        "Driver risk changed "
                        f"from {old_value} "
                        f"to {new_value}."
                    ),
                    data={
                        "old_risk":
                            old_value,

                        "new_risk":
                            new_value,

                        "driver_state":
                            normalize_value(
                                context
                                .driver
                                .state
                            ),
                    },
                )
            )

            # ------------------------------------------------
            # Entering HIGH risk is promoted into a dedicated
            # safety-relevant event.
            # ------------------------------------------------

            if (
                new_value
                == RiskLevel.HIGH.value
                and
                old_value
                != RiskLevel.HIGH.value
            ):

                events.append(
                    VehicleEvent(
                        type=(
                            EventType
                            .HIGH_RISK_DETECTED
                        ),
                        priority=(
                            EventPriority
                            .CRITICAL
                        ),
                        source=(
                            "cabin_perception"
                        ),
                        message=(
                            "High driver-risk "
                            "state detected."
                        ),
                        data={
                            "driver_state":
                                normalize_value(
                                    context
                                    .driver
                                    .state
                                ),

                            "risk":
                                new_value,

                            "perclos":
                                context
                                .driver
                                .perclos,

                            "eye_closure_seconds":
                                context
                                .driver
                                .eye_closure_seconds,

                            "recent_yawns":
                                context
                                .driver
                                .recent_yawns,

                            "vehicle_speed_kmh":
                                context
                                .vehicle
                                .speed_kmh,
                        },
                    )
                )

        # ----------------------------------------------------
        # Driver presence
        # ----------------------------------------------------

        elif (
            change.field
            == "presence"
        ):

            if (
                new_value
                == DriverPresence.ABSENT.value
            ):

                events.append(
                    VehicleEvent(
                        type=(
                            EventType
                            .DRIVER_ABSENT
                        ),
                        priority=(
                            EventPriority.HIGH
                        ),
                        source=(
                            "cabin_perception"
                        ),
                        message=(
                            "Driver is no longer "
                            "reliably detected."
                        ),
                        data={
                            "old_presence":
                                old_value,

                            "new_presence":
                                new_value,
                        },
                    )
                )

            elif (
                new_value
                == DriverPresence.PRESENT.value
            ):

                events.append(
                    VehicleEvent(
                        type=(
                            EventType
                            .DRIVER_PRESENT
                        ),
                        priority=(
                            EventPriority.LOW
                        ),
                        source=(
                            "cabin_perception"
                        ),
                        message=(
                            "Driver presence "
                            "confirmed."
                        ),
                        data={
                            "old_presence":
                                old_value,

                            "new_presence":
                                new_value,
                        },
                    )
                )

        return events

    # ========================================================
    # Road events
    # ========================================================

    def _road_events(
        self,
        change: ContextChange,
        context: VehicleContext,
    ) -> list[VehicleEvent]:

        events: list[
            VehicleEvent
        ] = []

        old_value = normalize_value(
            change.old_value
        )

        new_value = normalize_value(
            change.new_value
        )

        # ----------------------------------------------------
        # Traffic level
        # ----------------------------------------------------

        if (
            change.field
            == "traffic_level"
        ):

            events.append(
                VehicleEvent(
                    type=(
                        EventType
                        .TRAFFIC_LEVEL_CHANGED
                    ),
                    priority=(
                        EventPriority.MEDIUM
                    ),
                    source=(
                        "driving_perception"
                    ),
                    message=(
                        "Traffic level changed "
                        f"from {old_value} "
                        f"to {new_value}."
                    ),
                    data={
                        "old_level":
                            old_value,

                        "new_level":
                            new_value,

                        "vehicle_count":
                            context
                            .road
                            .vehicle_count,
                    },
                )
            )

        # ----------------------------------------------------
        # Lane lost
        # ----------------------------------------------------

        elif (
            change.field
            == "lane_detected"
            and
            change.old_value is True
            and
            change.new_value is False
        ):

            events.append(
                VehicleEvent(
                    type=(
                        EventType.LANE_LOST
                    ),
                    priority=(
                        EventPriority.HIGH
                    ),
                    source=(
                        "driving_perception"
                    ),
                    message=(
                        "Lane markings are "
                        "temporarily unavailable."
                    ),
                    data={
                        "lane_detected":
                            False,
                    },
                )
            )

        # ----------------------------------------------------
        # Drivable area lost
        # ----------------------------------------------------

        elif (
            change.field
            == "drivable_area_detected"
            and
            change.old_value is True
            and
            change.new_value is False
        ):

            events.append(
                VehicleEvent(
                    type=(
                        EventType
                        .DRIVABLE_AREA_LOST
                    ),
                    priority=(
                        EventPriority.HIGH
                    ),
                    source=(
                        "driving_perception"
                    ),
                    message=(
                        "Drivable area is "
                        "temporarily unavailable."
                    ),
                    data={
                        "drivable_area_detected":
                            False,
                    },
                )
            )

        return events

    # ========================================================
    # Vehicle events
    # ========================================================

    def _vehicle_events(
        self,
        change: ContextChange,
        context: VehicleContext,
    ) -> list[VehicleEvent]:

        events: list[
            VehicleEvent
        ] = []

        old_value = normalize_value(
            change.old_value
        )

        new_value = normalize_value(
            change.new_value
        )

        # ----------------------------------------------------
        # Navigation state
        # ----------------------------------------------------

        if (
            change.field
            == "navigation_state"
        ):

            events.append(
                VehicleEvent(
                    type=(
                        EventType
                        .NAVIGATION_STATE_CHANGED
                    ),
                    priority=(
                        EventPriority.MEDIUM
                    ),
                    source="vehicle_state",
                    message=(
                        "Navigation state changed "
                        f"from {old_value} "
                        f"to {new_value}."
                    ),
                    data={
                        "old_state":
                            old_value,

                        "new_state":
                            new_value,

                        "destination":
                            context
                            .vehicle
                            .navigation_destination,
                    },
                )
            )

        # ----------------------------------------------------
        # Notice:
        #
        # speed changes intentionally produce NO semantic
        # event here.
        #
        # 67.9 -> 68.0 km/h should not wake up an LLM.
        # ----------------------------------------------------

        return events

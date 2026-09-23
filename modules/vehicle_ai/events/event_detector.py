from __future__ import annotations

from typing import Any

from modules.config.events import EventTimingConfig
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
from modules.vehicle_ai.events.temporal_gate import StableValueGate, TemporalGate


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
        return str(value.value)

    return str(value)


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

    def __init__(self, timing: EventTimingConfig | None = None) -> None:
        policy = timing or EventTimingConfig.load()
        self._risk_gate = TemporalGate(**vars(policy.high_driver_risk))
        self._risk_state = StableValueGate(
            initial=RiskLevel.UNKNOWN.value,
            hold_ms=policy.high_driver_risk.hold_ms,
            max_gap_ms=policy.high_driver_risk.max_gap_ms,
        )
        self._lane_gate = TemporalGate(**vars(policy.lane_lost))
        self._area_gate = TemporalGate(**vars(policy.drivable_area_lost))
        self._seen_lane = False
        self._seen_area = False

    def invalidate_observation(self, domain: str) -> None:
        """An invalid frame breaks evidence continuity without changing context."""

        if domain == "driver":
            self._risk_gate.reset()
            self._risk_state.reset()
        elif domain == "road":
            self._lane_gate.reset()
            self._area_gate.reset()

    def observe_hazards(
        self, domain: str, context: VehicleContext, *, at_ms: int
    ) -> list[VehicleEvent]:
        """Evaluate alerts on every valid observation, even unchanged ones."""

        events: list[VehicleEvent] = []
        if domain == "driver":
            risk = normalize_value(context.driver.risk)
            transition = self._risk_state.observe(risk, at_ms=at_ms)
            if transition is not None:
                old_risk, new_risk = transition
                priority = (
                    EventPriority.HIGH
                    if new_risk == RiskLevel.HIGH.value
                    else EventPriority.MEDIUM
                    if new_risk == RiskLevel.MEDIUM.value
                    else EventPriority.LOW
                )
                events.append(
                    VehicleEvent(
                        type=EventType.DRIVER_RISK_CHANGED,
                        priority=priority,
                        source="cabin_perception",
                        message=f"Driver risk changed from {old_risk} to {new_risk}.",
                        data={
                            "old_risk": old_risk,
                            "new_risk": new_risk,
                            "driver_state": normalize_value(context.driver.state),
                        },
                    )
                )
            if self._risk_gate.observe(
                risk == RiskLevel.HIGH.value,
                at_ms=at_ms,
            ):
                events.append(
                    VehicleEvent(
                        type=EventType.HIGH_RISK_DETECTED,
                        priority=EventPriority.CRITICAL,
                        source="cabin_perception",
                        message="High driver-risk state detected.",
                        data={
                            "driver_state": normalize_value(context.driver.state),
                            "risk": RiskLevel.HIGH.value,
                            "perclos": context.driver.perclos,
                            "eye_closure_seconds": context.driver.eye_closure_seconds,
                            "recent_yawns": context.driver.recent_yawns,
                            "vehicle_speed_kmh": context.vehicle.speed_kmh,
                        },
                    )
                )
        elif domain == "road":
            lane = context.road.lane_detected
            area = context.road.drivable_area_detected
            if lane:
                self._seen_lane = True
            if area:
                self._seen_area = True
            if self._lane_gate.observe(self._seen_lane and not lane, at_ms=at_ms):
                events.append(
                    VehicleEvent(
                        type=EventType.LANE_LOST,
                        priority=EventPriority.HIGH,
                        source="driving_perception",
                        message="Lane markings are temporarily unavailable.",
                        data={"lane_detected": False},
                    )
                )
            if self._area_gate.observe(self._seen_area and not area, at_ms=at_ms):
                events.append(
                    VehicleEvent(
                        type=EventType.DRIVABLE_AREA_LOST,
                        priority=EventPriority.HIGH,
                        source="driving_perception",
                        message="Drivable area is temporarily unavailable.",
                        data={"drivable_area_detected": False},
                    )
                )
        return events

    def detect(
        self,
        changes: list[ContextChange],
        context: VehicleContext,
    ) -> list[VehicleEvent]:

        events: list[VehicleEvent] = []

        for change in changes:
            if change.domain == ContextDomain.DRIVER:
                events.extend(
                    self._driver_events(
                        change,
                        context,
                    )
                )

            elif change.domain == ContextDomain.ROAD:
                events.extend(
                    self._road_events(
                        change,
                        context,
                    )
                )

            elif change.domain == ContextDomain.VEHICLE:
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

        events: list[VehicleEvent] = []

        old_value = normalize_value(change.old_value)

        new_value = normalize_value(change.new_value)

        # ----------------------------------------------------
        # Driver state transition
        # ----------------------------------------------------

        if change.field == "state":
            if new_value == DriverState.DROWSY.value:
                priority = EventPriority.HIGH

            elif new_value == DriverState.SUSPECTED.value:
                priority = EventPriority.MEDIUM

            else:
                priority = EventPriority.LOW

            events.append(
                VehicleEvent(
                    type=(EventType.DRIVER_STATE_CHANGED),
                    priority=priority,
                    source="cabin_perception",
                    message=(f"Driver state changed from {old_value} to {new_value}."),
                    data={
                        "old_state": old_value,
                        "new_state": new_value,
                        "risk": normalize_value(context.driver.risk),
                        "perclos": context.driver.perclos,
                        "eye_closure_seconds": context.driver.eye_closure_seconds,
                        "recent_yawns": context.driver.recent_yawns,
                    },
                )
            )

        # ----------------------------------------------------
        # Driver presence
        # ----------------------------------------------------

        elif change.field == "presence":
            if new_value == DriverPresence.ABSENT.value:
                events.append(
                    VehicleEvent(
                        type=(EventType.DRIVER_ABSENT),
                        priority=(EventPriority.HIGH),
                        source=("cabin_perception"),
                        message=("Driver is no longer reliably detected."),
                        data={
                            "old_presence": old_value,
                            "new_presence": new_value,
                        },
                    )
                )

            elif new_value == DriverPresence.PRESENT.value:
                events.append(
                    VehicleEvent(
                        type=(EventType.DRIVER_PRESENT),
                        priority=(EventPriority.LOW),
                        source=("cabin_perception"),
                        message=("Driver presence confirmed."),
                        data={
                            "old_presence": old_value,
                            "new_presence": new_value,
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

        events: list[VehicleEvent] = []

        old_value = normalize_value(change.old_value)

        new_value = normalize_value(change.new_value)

        # ----------------------------------------------------
        # Traffic level
        # ----------------------------------------------------

        if change.field == "traffic_level":
            events.append(
                VehicleEvent(
                    type=(EventType.TRAFFIC_LEVEL_CHANGED),
                    priority=(EventPriority.MEDIUM),
                    source=("driving_perception"),
                    message=(f"Traffic level changed from {old_value} to {new_value}."),
                    data={
                        "old_level": old_value,
                        "new_level": new_value,
                        "vehicle_count": context.road.vehicle_count,
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

        events: list[VehicleEvent] = []

        old_value = normalize_value(change.old_value)

        new_value = normalize_value(change.new_value)

        # ----------------------------------------------------
        # Navigation state
        # ----------------------------------------------------

        if change.field == "navigation_state":
            events.append(
                VehicleEvent(
                    type=(EventType.NAVIGATION_STATE_CHANGED),
                    priority=(EventPriority.MEDIUM),
                    source="vehicle_state",
                    message=(
                        f"Navigation state changed from {old_value} to {new_value}."
                    ),
                    data={
                        "old_state": old_value,
                        "new_state": new_value,
                        "destination": context.vehicle.navigation_destination,
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

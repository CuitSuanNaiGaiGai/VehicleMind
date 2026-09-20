from __future__ import annotations

from typing import Any

from modules.vehicle_ai.context import (
    ContextManager,
    DriverPresence,
    DriverState,
    RiskLevel,
)


class CabinContextAdapter:
    """
    Bridge Cabin Intelligence semantic outputs into
    VehicleMind DriverContext.

    Important:
        Driver presence must come from DriverPresenceTracker.

        face_visible is only the current-frame observation and
        must NOT be treated as the semantic driver-presence
        state.
    """

    def __init__(
        self,
        context_manager: ContextManager,
    ):
        self.context_manager = (
            context_manager
        )

    # ========================================================
    # Enum helpers
    # ========================================================

    @staticmethod
    def _enum_text(
        value: Any,
    ) -> str:

        if value is None:
            return "UNKNOWN"

        if hasattr(
            value,
            "value",
        ):
            return str(
                value.value
            )

        if hasattr(
            value,
            "name",
        ):
            return str(
                value.name
            )

        return str(
            value
        )

    # ========================================================
    # Normalize Presence
    # ========================================================

    def _presence(
        self,
        value: Any,
    ) -> DriverPresence:

        text = (
            self._enum_text(
                value
            )
            .upper()
        )

        try:
            return DriverPresence(
                text
            )

        except ValueError:
            return (
                DriverPresence.UNKNOWN
            )

    # ========================================================
    # Normalize Driver State
    # ========================================================

    def _driver_state(
        self,
        value: Any,
    ) -> DriverState:

        text = (
            self._enum_text(
                value
            )
            .upper()
        )

        try:
            return DriverState(
                text
            )

        except ValueError:
            return (
                DriverState.UNKNOWN
            )

    # ========================================================
    # Normalize Risk
    # ========================================================

    def _risk_level(
        self,
        value: Any,
    ) -> RiskLevel:

        text = (
            self._enum_text(
                value
            )
            .upper()
        )

        try:
            return RiskLevel(
                text
            )

        except ValueError:
            return (
                RiskLevel.UNKNOWN
            )

    # ========================================================
    # Update
    # ========================================================

    def update(
        self,
        *,
        presence: Any,
        driver_state: Any,
        risk: Any,
        perclos: float | None = None,
        eye_closed: bool | None = None,
        eye_closure_seconds: float = 0.0,
        recent_yawns: int = 0,
        blink_count: int = 0,
    ):
        """
        Update DriverContext using semantic outputs from
        CabinPerceptionService.
        """

        return (
            self.context_manager
            .update_driver(
                presence=(
                    self._presence(
                        presence
                    )
                ),
                state=(
                    self._driver_state(
                        driver_state
                    )
                ),
                risk=(
                    self._risk_level(
                        risk
                    )
                ),
                perclos=perclos,
                eye_closed=(
                    None
                    if eye_closed is None
                    else bool(eye_closed)
                ),
                eye_closure_seconds=float(
                    eye_closure_seconds
                ),
                recent_yawns=int(
                    recent_yawns
                ),
                blink_count=int(
                    blink_count
                ),
            )
        )
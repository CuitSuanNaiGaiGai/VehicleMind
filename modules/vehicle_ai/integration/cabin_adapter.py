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
    Bridge Cabin Intelligence outputs into VehicleContext.

    This layer only translates semantic outputs.
    It does NOT perform fatigue estimation itself.
    """

    def __init__(
        self,
        context_manager: ContextManager,
    ):
        self.context_manager = context_manager

    # ========================================================
    # Enum normalization
    # ========================================================

    @staticmethod
    def _enum_text(
        value: Any,
    ) -> str:

        if value is None:
            return "UNKNOWN"

        if hasattr(value, "value"):
            return str(value.value)

        if hasattr(value, "name"):
            return str(value.name)

        return str(value)

    def _driver_state(
        self,
        value: Any,
    ) -> DriverState:

        text = (
            self._enum_text(value)
            .upper()
        )

        try:
            return DriverState(text)

        except ValueError:
            return DriverState.UNKNOWN

    def _risk_level(
        self,
        value: Any,
    ) -> RiskLevel:

        if value is None:
            return RiskLevel.UNKNOWN

        text = (
            self._enum_text(value)
            .upper()
        )

        try:
            return RiskLevel(text)

        except ValueError:
            return RiskLevel.UNKNOWN

    # ========================================================
    # Update
    # ========================================================

    def update(
        self,
        *,
        face_present: bool,
        driver_state: Any,
        risk: Any = None,
        perclos: float | None = None,
        eye_closed: bool = False,
        eye_closure_seconds: float = 0.0,
        recent_yawns: int = 0,
        blink_count: int = 0,
    ):
        """
        Update DriverContext from one Cabin Intelligence frame.
        """

        presence = (
            DriverPresence.PRESENT
            if face_present
            else DriverPresence.ABSENT
        )

        state = (
            self._driver_state(
                driver_state
            )
        )

        risk_level = (
            self._risk_level(
                risk
            )
        )

        return (
            self.context_manager
            .update_driver(
                presence=presence,
                state=state,
                risk=risk_level,
                perclos=perclos,
                eye_closed=bool(
                    eye_closed
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
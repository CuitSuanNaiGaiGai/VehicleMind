from __future__ import annotations

from typing import Any

from modules.observation import ObservationMetadata
from modules.vehicle_ai.context import (
    ContextManager,
    DriverPresence,
    DriverState,
    RiskLevel,
)


_UNSET = object()


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
        self.context_manager = context_manager

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
            return str(value.value)

        if hasattr(
            value,
            "name",
        ):
            return str(value.name)

        return str(value)

    # ========================================================
    # Normalize Presence
    # ========================================================

    def _presence(
        self,
        value: Any,
    ) -> DriverPresence:

        text = self._enum_text(value).upper()

        try:
            return DriverPresence(text)

        except ValueError:
            return DriverPresence.UNKNOWN

    # ========================================================
    # Normalize Driver State
    # ========================================================

    def _driver_state(
        self,
        value: Any,
    ) -> DriverState:

        text = self._enum_text(value).upper()

        try:
            return DriverState(text)

        except ValueError:
            return DriverState.UNKNOWN

    # ========================================================
    # Normalize Risk
    # ========================================================

    def _risk_level(
        self,
        value: Any,
    ) -> RiskLevel:

        text = self._enum_text(value).upper()

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
        observation: ObservationMetadata | None = None,
        presence: Any,
        driver_state: Any,
        risk: Any,
        perclos: Any = _UNSET,
        eye_closed: Any = _UNSET,
        eye_closure_seconds: Any = _UNSET,
        recent_yawns: Any = _UNSET,
        blink_count: Any = _UNSET,
    ):
        """
        Update DriverContext using semantic outputs from
        CabinPerceptionService.
        """

        updates: dict[str, Any] = {
            "presence": self._presence(presence),
            "state": self._driver_state(driver_state),
            "risk": self._risk_level(risk),
        }
        if perclos is not _UNSET:
            updates["perclos"] = perclos
        if eye_closed is not _UNSET:
            updates["eye_closed"] = None if eye_closed is None else bool(eye_closed)
        if eye_closure_seconds is not _UNSET:
            updates["eye_closure_seconds"] = float(eye_closure_seconds)
        if recent_yawns is not _UNSET:
            updates["recent_yawns"] = int(recent_yawns)
        if blink_count is not _UNSET:
            updates["blink_count"] = int(blink_count)
        return self.context_manager.update_driver(observation=observation, **updates)

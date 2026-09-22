from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from modules.cabin.presence.driver_presence import (
    DriverPresence,
)


class DriverState(str, Enum):
    UNKNOWN = "UNKNOWN"
    WARMING_UP = "WARMING_UP"
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    DROWSY = "DROWSY"


class RiskLevel(str, Enum):
    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class DriverStateResult:
    """
    High-level driver state.

    Intermediate fatigue signals are retained for
    visualization and debugging.
    """

    state: DriverState

    risk_level: RiskLevel

    continuous_eye_closure: float

    perclos: float

    recent_yawns: int

    perclos_ready: bool

    reason: str


class DriverStateEstimator:
    """
    Lightweight multi-cue driver-state estimator.

    Inputs:
        - Driver Presence
        - Continuous eye closure
        - PERCLOS
        - Recent yawns

    Output:
        UNKNOWN
        WARMING_UP
        NORMAL
        SUSPECTED
        DROWSY

    The thresholds are engineering parameters for
    demonstration purposes, not medical or production
    DMS standards.
    """

    def __init__(
        self,
        suspected_perclos: float = 0.25,
        drowsy_perclos: float = 0.30,
        suspected_closure_seconds: float = 1.2,
        drowsy_closure_seconds: float = 2.0,
        yawn_window_seconds: float = 60.0,
        suspected_yawns: int = 2,
    ):
        self.suspected_perclos = suspected_perclos
        self.drowsy_perclos = drowsy_perclos

        self.suspected_closure_seconds = suspected_closure_seconds

        self.drowsy_closure_seconds = drowsy_closure_seconds

        self.yawn_window_ms = int(yawn_window_seconds * 1000)

        self.suspected_yawns = suspected_yawns

        # -----------------------------------------------------
        # Eye closure tracking
        # -----------------------------------------------------

        self._eye_closed_since_ms: Optional[int] = None

        # -----------------------------------------------------
        # Yawn event tracking
        # -----------------------------------------------------

        self._yawn_timestamps: Deque[int] = deque()

        self._last_yawn_count = 0

    # =========================================================
    # Eye closure
    # =========================================================

    def _update_eye_closure(
        self,
        timestamp_ms: int,
        eye_closed: bool,
    ) -> float:

        if eye_closed:
            if self._eye_closed_since_ms is None:
                self._eye_closed_since_ms = timestamp_ms

            duration_ms = timestamp_ms - self._eye_closed_since_ms

            return duration_ms / 1000.0

        self._eye_closed_since_ms = None

        return 0.0

    # =========================================================
    # Yawn history
    # =========================================================

    def _update_yawns(
        self,
        timestamp_ms: int,
        yawn_count: int,
    ) -> int:

        # -----------------------------------------------------
        # Detect newly completed yawn events
        # -----------------------------------------------------

        if yawn_count > self._last_yawn_count:
            new_events = yawn_count - self._last_yawn_count

            for _ in range(new_events):
                self._yawn_timestamps.append(timestamp_ms)

        self._last_yawn_count = yawn_count

        # -----------------------------------------------------
        # Remove events outside the rolling window
        # -----------------------------------------------------

        cutoff = timestamp_ms - self.yawn_window_ms

        while self._yawn_timestamps and self._yawn_timestamps[0] < cutoff:
            self._yawn_timestamps.popleft()

        return len(self._yawn_timestamps)

    # =========================================================
    # Reset transient fatigue observations
    # =========================================================

    def _reset_transient_state(
        self,
    ) -> None:

        self._eye_closed_since_ms = None

    # =========================================================
    # Main update
    # =========================================================

    def update(
        self,
        timestamp_ms: int,
        driver_presence: DriverPresence,
        eye_closed: bool,
        perclos: float,
        perclos_ready: bool,
        yawn_count: int,
    ) -> DriverStateResult:

        # =====================================================
        # 1. Driver not reliably observable
        # =====================================================

        if driver_presence != DriverPresence.PRESENT:
            self._reset_transient_state()

            return DriverStateResult(
                state=DriverState.UNKNOWN,
                risk_level=RiskLevel.UNKNOWN,
                continuous_eye_closure=0.0,
                perclos=perclos,
                recent_yawns=0,
                perclos_ready=perclos_ready,
                reason="driver_not_present",
            )

        # =====================================================
        # 2. Update fatigue evidence
        # =====================================================

        closure_duration = self._update_eye_closure(
            timestamp_ms=timestamp_ms,
            eye_closed=eye_closed,
        )

        recent_yawns = self._update_yawns(
            timestamp_ms=timestamp_ms,
            yawn_count=yawn_count,
        )

        # =====================================================
        # 3. Strong fatigue evidence
        # =====================================================

        if closure_duration >= self.drowsy_closure_seconds:
            return DriverStateResult(
                state=DriverState.DROWSY,
                risk_level=RiskLevel.HIGH,
                continuous_eye_closure=closure_duration,
                perclos=perclos,
                recent_yawns=recent_yawns,
                perclos_ready=perclos_ready,
                reason="prolonged_eye_closure",
            )

        if perclos_ready and perclos >= self.drowsy_perclos:
            return DriverStateResult(
                state=DriverState.DROWSY,
                risk_level=RiskLevel.HIGH,
                continuous_eye_closure=closure_duration,
                perclos=perclos,
                recent_yawns=recent_yawns,
                perclos_ready=perclos_ready,
                reason="high_perclos",
            )

        # -----------------------------------------------------
        # Combined evidence:
        #
        # moderately elevated PERCLOS + repeated yawns
        # -----------------------------------------------------

        if (
            perclos_ready
            and perclos >= self.suspected_perclos
            and recent_yawns >= self.suspected_yawns
        ):
            return DriverStateResult(
                state=DriverState.DROWSY,
                risk_level=RiskLevel.HIGH,
                continuous_eye_closure=closure_duration,
                perclos=perclos,
                recent_yawns=recent_yawns,
                perclos_ready=perclos_ready,
                reason="perclos_and_repeated_yawn",
            )

        # =====================================================
        # 4. Suspected fatigue
        # =====================================================

        if closure_duration >= self.suspected_closure_seconds:
            return DriverStateResult(
                state=DriverState.SUSPECTED,
                risk_level=RiskLevel.MEDIUM,
                continuous_eye_closure=closure_duration,
                perclos=perclos,
                recent_yawns=recent_yawns,
                perclos_ready=perclos_ready,
                reason="extended_eye_closure",
            )

        if perclos_ready and perclos >= self.suspected_perclos:
            return DriverStateResult(
                state=DriverState.SUSPECTED,
                risk_level=RiskLevel.MEDIUM,
                continuous_eye_closure=closure_duration,
                perclos=perclos,
                recent_yawns=recent_yawns,
                perclos_ready=perclos_ready,
                reason="elevated_perclos",
            )

        if recent_yawns >= self.suspected_yawns:
            return DriverStateResult(
                state=DriverState.SUSPECTED,
                risk_level=RiskLevel.MEDIUM,
                continuous_eye_closure=closure_duration,
                perclos=perclos,
                recent_yawns=recent_yawns,
                perclos_ready=perclos_ready,
                reason="repeated_yawn",
            )

        # =====================================================
        # 5. PERCLOS is still accumulating
        # =====================================================

        if not perclos_ready:
            return DriverStateResult(
                state=DriverState.WARMING_UP,
                risk_level=RiskLevel.UNKNOWN,
                continuous_eye_closure=closure_duration,
                perclos=perclos,
                recent_yawns=recent_yawns,
                perclos_ready=False,
                reason="perclos_warming_up",
            )

        # =====================================================
        # 6. Normal
        # =====================================================

        return DriverStateResult(
            state=DriverState.NORMAL,
            risk_level=RiskLevel.LOW,
            continuous_eye_closure=closure_duration,
            perclos=perclos,
            recent_yawns=recent_yawns,
            perclos_ready=True,
            reason="normal",
        )

    # =========================================================
    # Reset
    # =========================================================

    def reset(
        self,
    ) -> None:

        self._eye_closed_since_ms = None

        self._yawn_timestamps.clear()

        self._last_yawn_count = 0

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DriverState(str, Enum):
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
    state: DriverState
    risk_level: RiskLevel

    perclos: float
    continuous_eye_closure: float

    reason: str


class DriverStateEstimator:
    """
    First-stage driver fatigue state estimator.

    This version combines two temporal cues:

    1. Continuous eye-closure duration
    2. Sliding-window PERCLOS

    The thresholds are currently demonstration heuristics.
    They will later be calibrated/evaluated with public DMS datasets.
    """

    def __init__(
        self,
        suspected_perclos: float = 0.15,
        drowsy_perclos: float = 0.30,
        suspected_closure_seconds: float = 1.2,
        drowsy_closure_seconds: float = 2.5,
    ):
        self.suspected_perclos = suspected_perclos
        self.drowsy_perclos = drowsy_perclos

        self.suspected_closure_ms = int(
            suspected_closure_seconds * 1000
        )

        self.drowsy_closure_ms = int(
            drowsy_closure_seconds * 1000
        )

        self._eye_closed_since_ms: Optional[int] = None

    def update(
        self,
        timestamp_ms: int,
        eye_closed: Optional[bool],
        perclos: float,
        perclos_ready: bool,
    ) -> DriverStateResult:

        #
        # Track continuous eye closure
        #

        if eye_closed is True:

            if self._eye_closed_since_ms is None:
                self._eye_closed_since_ms = timestamp_ms

        else:
            self._eye_closed_since_ms = None

        if self._eye_closed_since_ms is not None:

            closure_ms = (
                timestamp_ms
                - self._eye_closed_since_ms
            )

        else:
            closure_ms = 0

        closure_seconds = closure_ms / 1000.0

        #
        # HIGH RISK:
        # unusually long continuous eye closure
        #

        if closure_ms >= self.drowsy_closure_ms:

            return DriverStateResult(
                state=DriverState.DROWSY,
                risk_level=RiskLevel.HIGH,
                perclos=perclos,
                continuous_eye_closure=closure_seconds,
                reason="Prolonged eye closure detected",
            )

        #
        # HIGH RISK:
        # high PERCLOS
        #

        if (
            perclos_ready
            and perclos >= self.drowsy_perclos
        ):

            return DriverStateResult(
                state=DriverState.DROWSY,
                risk_level=RiskLevel.HIGH,
                perclos=perclos,
                continuous_eye_closure=closure_seconds,
                reason="High PERCLOS detected",
            )

        #
        # MEDIUM RISK
        #

        if closure_ms >= self.suspected_closure_ms:

            return DriverStateResult(
                state=DriverState.SUSPECTED,
                risk_level=RiskLevel.MEDIUM,
                perclos=perclos,
                continuous_eye_closure=closure_seconds,
                reason="Extended eye closure",
            )

        if (
            perclos_ready
            and perclos >= self.suspected_perclos
        ):

            return DriverStateResult(
                state=DriverState.SUSPECTED,
                risk_level=RiskLevel.MEDIUM,
                perclos=perclos,
                continuous_eye_closure=closure_seconds,
                reason="Elevated PERCLOS",
            )

        #
        # Warm-up
        #

        if not perclos_ready:

            return DriverStateResult(
                state=DriverState.WARMING_UP,
                risk_level=RiskLevel.UNKNOWN,
                perclos=perclos,
                continuous_eye_closure=closure_seconds,
                reason="Collecting temporal observations",
            )

        #
        # Normal
        #

        return DriverStateResult(
            state=DriverState.NORMAL,
            risk_level=RiskLevel.LOW,
            perclos=perclos,
            continuous_eye_closure=closure_seconds,
            reason="Driver condition normal",
        )

    def reset(self) -> None:
        self._eye_closed_since_ms = None

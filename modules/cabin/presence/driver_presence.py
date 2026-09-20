from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DriverPresence(str, Enum):
    """
    Stable driver-presence state exposed to upper-level modules.
    """

    UNKNOWN = "UNKNOWN"
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


@dataclass
class DriverPresenceResult:
    """
    Stable driver-presence result.

    state:
        UNKNOWN / PRESENT / ABSENT

    detected_this_frame:
        Whether a valid driver face was detected in the current frame.

    lost_duration:
        Number of seconds since the last valid face observation.
        Zero when a face is currently visible.

    visible_duration:
        Continuous confirmed face-visible duration.
    """

    state: DriverPresence

    detected_this_frame: bool

    lost_duration: float

    visible_duration: float


class DriverPresenceTracker:
    """
    Temporal smoothing for driver presence.

    The face detector may occasionally miss one or several frames.
    Therefore:

        one missed frame != driver absent

    Logic:

        Face detected continuously
            -> PRESENT

        Face briefly lost
            -> keep PRESENT

        Face missing longer than absence_timeout
            -> ABSENT

        At system startup, if no face has ever been observed
        for startup_timeout
            -> ABSENT
    """

    def __init__(
        self,
        present_confirm_seconds: float = 0.15,
        absence_timeout_seconds: float = 1.5,
        startup_timeout_seconds: float = 1.0,
    ):
        self.present_confirm_ms = int(
            present_confirm_seconds * 1000
        )

        self.absence_timeout_ms = int(
            absence_timeout_seconds * 1000
        )

        self.startup_timeout_ms = int(
            startup_timeout_seconds * 1000
        )

        self._state = DriverPresence.UNKNOWN

        self._start_timestamp_ms: Optional[int] = None

        self._last_seen_timestamp_ms: Optional[int] = None

        self._visible_since_ms: Optional[int] = None

        self._present_candidate_since_ms: Optional[int] = None

    @property
    def state(self) -> DriverPresence:
        return self._state

    def update(
        self,
        timestamp_ms: int,
        face_detected: bool,
    ) -> DriverPresenceResult:

        # -----------------------------------------------------
        # Initialize tracker timestamp
        # -----------------------------------------------------

        if self._start_timestamp_ms is None:
            self._start_timestamp_ms = timestamp_ms

        # =====================================================
        # Face detected
        # =====================================================

        if face_detected:

            self._last_seen_timestamp_ms = timestamp_ms

            # Start candidate-present period.
            if self._present_candidate_since_ms is None:
                self._present_candidate_since_ms = timestamp_ms

            # If already PRESENT, remain PRESENT immediately.
            if self._state == DriverPresence.PRESENT:

                if self._visible_since_ms is None:
                    self._visible_since_ms = timestamp_ms

            else:

                present_duration_ms = (
                    timestamp_ms
                    - self._present_candidate_since_ms
                )

                if (
                    present_duration_ms
                    >= self.present_confirm_ms
                ):
                    self._state = DriverPresence.PRESENT

                    self._visible_since_ms = (
                        self._present_candidate_since_ms
                    )

            lost_duration = 0.0

            if self._visible_since_ms is not None:
                visible_duration = (
                    timestamp_ms
                    - self._visible_since_ms
                ) / 1000.0
            else:
                visible_duration = 0.0

            return DriverPresenceResult(
                state=self._state,
                detected_this_frame=True,
                lost_duration=lost_duration,
                visible_duration=visible_duration,
            )

        # =====================================================
        # Face NOT detected
        # =====================================================

        self._present_candidate_since_ms = None

        # -----------------------------------------------------
        # Driver has previously been visible
        # -----------------------------------------------------

        if self._last_seen_timestamp_ms is not None:

            lost_ms = (
                timestamp_ms
                - self._last_seen_timestamp_ms
            )

            lost_duration = (
                lost_ms / 1000.0
            )

            # Short detector dropout:
            # retain PRESENT.
            if (
                self._state == DriverPresence.PRESENT
                and
                lost_ms <= self.absence_timeout_ms
            ):

                if self._visible_since_ms is not None:

                    visible_duration = (
                        self._last_seen_timestamp_ms
                        - self._visible_since_ms
                    ) / 1000.0

                else:
                    visible_duration = 0.0

                return DriverPresenceResult(
                    state=DriverPresence.PRESENT,
                    detected_this_frame=False,
                    lost_duration=lost_duration,
                    visible_duration=visible_duration,
                )

            # Missing for long enough -> ABSENT.
            if lost_ms > self.absence_timeout_ms:

                self._state = DriverPresence.ABSENT

                self._visible_since_ms = None

            return DriverPresenceResult(
                state=self._state,
                detected_this_frame=False,
                lost_duration=lost_duration,
                visible_duration=0.0,
            )

        # -----------------------------------------------------
        # Driver has never been detected since startup
        # -----------------------------------------------------

        startup_elapsed_ms = (
            timestamp_ms
            - self._start_timestamp_ms
        )

        if (
            startup_elapsed_ms
            >= self.startup_timeout_ms
        ):
            self._state = DriverPresence.ABSENT

        return DriverPresenceResult(
            state=self._state,
            detected_this_frame=False,
            lost_duration=0.0,
            visible_duration=0.0,
        )

    def reset(self) -> None:

        self._state = DriverPresence.UNKNOWN

        self._start_timestamp_ms = None

        self._last_seen_timestamp_ms = None

        self._visible_since_ms = None

        self._present_candidate_since_ms = None

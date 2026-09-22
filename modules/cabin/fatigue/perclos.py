from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerclosResult:
    perclos: float
    closed_duration: float
    observed_duration: float
    ready: bool


class PerclosEstimator:
    """
    Time-weighted PERCLOS estimator.

    Instead of using:
        closed_frames / total_frames

    we calculate:
        closed_time / valid_observation_time

    so the result is less sensitive to FPS fluctuation.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        min_observation_seconds: float = 5.0,
    ):
        self.window_ms = int(window_seconds * 1000)
        self.min_observation_ms = int(min_observation_seconds * 1000)

        # Each element:
        # (start_ms, end_ms, eye_closed)
        #
        # eye_closed:
        #   True  -> closed
        #   False -> open
        #   None  -> invalid / face missing
        self._segments = deque()

        self._last_timestamp_ms = None
        self._last_state: Optional[bool] = None

    def update(
        self,
        timestamp_ms: int,
        is_closed: Optional[bool],
    ) -> PerclosResult:

        if self._last_timestamp_ms is not None:
            if timestamp_ms <= self._last_timestamp_ms:
                raise ValueError("timestamp_ms must be monotonically increasing")

            self._segments.append(
                (
                    self._last_timestamp_ms,
                    timestamp_ms,
                    self._last_state,
                )
            )

        self._last_timestamp_ms = timestamp_ms
        self._last_state = is_closed

        cutoff = timestamp_ms - self.window_ms

        #
        # Remove segments completely outside the window.
        #
        while self._segments and self._segments[0][1] <= cutoff:
            self._segments.popleft()

        #
        # Trim the first segment if only part of it
        # remains inside the window.
        #
        if self._segments:
            start_ms, end_ms, state = self._segments[0]

            if start_ms < cutoff < end_ms:
                self._segments[0] = (
                    cutoff,
                    end_ms,
                    state,
                )

        closed_ms = 0
        observed_ms = 0

        for start_ms, end_ms, state in self._segments:
            duration_ms = end_ms - start_ms

            # Face/eye information unavailable.
            if state is None:
                continue

            observed_ms += duration_ms

            if state:
                closed_ms += duration_ms

        if observed_ms > 0:
            perclos = closed_ms / observed_ms
        else:
            perclos = 0.0

        ready = observed_ms >= self.min_observation_ms

        return PerclosResult(
            perclos=perclos,
            closed_duration=closed_ms / 1000.0,
            observed_duration=observed_ms / 1000.0,
            ready=ready,
        )

    def reset(self) -> None:

        self._segments.clear()

        self._last_timestamp_ms = None
        self._last_state = None

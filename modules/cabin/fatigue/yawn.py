from dataclasses import dataclass


@dataclass
class YawnResult:
    yawn_count: int
    open_duration: float
    is_yawning: bool
    yawn_event: bool


class YawnDetector:
    """
    Temporal yawn detector.

    A yawn is detected when the mouth remains open
    longer than min_open_seconds.
    """

    def __init__(
        self,
        min_open_seconds: float = 1.2,
    ):
        self.min_open_ms = int(
            min_open_seconds * 1000
        )

        self._open_start_ms = None

        self._event_triggered = False

        self.yawn_count = 0

    def update(
        self,
        timestamp_ms: int,
        mouth_open: bool,
    ) -> YawnResult:

        yawn_event = False
        open_duration_ms = 0

        if mouth_open:

            if self._open_start_ms is None:
                self._open_start_ms = timestamp_ms

            open_duration_ms = (
                timestamp_ms
                - self._open_start_ms
            )

            if (
                open_duration_ms >= self.min_open_ms
                and not self._event_triggered
            ):

                self.yawn_count += 1

                self._event_triggered = True

                yawn_event = True

        else:

            self._open_start_ms = None

            self._event_triggered = False

        return YawnResult(
            yawn_count=self.yawn_count,
            open_duration=(
                open_duration_ms / 1000.0
            ),
            is_yawning=(
                mouth_open
                and open_duration_ms
                >= self.min_open_ms
            ),
            yawn_event=yawn_event,
        )

    def reset(self) -> None:

        self._open_start_ms = None

        self._event_triggered = False

        self.yawn_count = 0

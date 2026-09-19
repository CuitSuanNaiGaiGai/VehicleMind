from dataclasses import dataclass


@dataclass
class BlinkResult:
    blink_count: int
    closed_frames: int
    is_blinking: bool


class BlinkDetector:
    """
    Temporal blink detector based on consecutive closed-eye frames.

    A blink is counted when:
        OPEN -> CLOSED for several frames -> OPEN
    """

    def __init__(
        self,
        min_closed_frames: int = 2,
        max_closed_frames: int = 15,
    ):
        self.min_closed_frames = min_closed_frames
        self.max_closed_frames = max_closed_frames

        self.closed_frames = 0
        self.blink_count = 0

    def update(
        self,
        is_closed: bool,
    ) -> BlinkResult:

        is_blinking = False

        if is_closed:

            self.closed_frames += 1

        else:

            if (
                self.min_closed_frames
                <= self.closed_frames
                <= self.max_closed_frames
            ):
                self.blink_count += 1
                is_blinking = True

            self.closed_frames = 0

        return BlinkResult(
            blink_count=self.blink_count,
            closed_frames=self.closed_frames,
            is_blinking=is_blinking,
        )

    def reset(self) -> None:
        self.closed_frames = 0
        self.blink_count = 0

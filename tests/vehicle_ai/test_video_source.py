from __future__ import annotations

import numpy as np

from apps.vehicle_ai_demo.video_source import VideoFileSource


class _Capture:
    def __init__(self) -> None:
        self.frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        import cv2

        if property_id == cv2.CAP_PROP_FPS:
            return 2.0
        if property_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def set(self, property_id: int, value: float) -> bool:
        import cv2

        if property_id == cv2.CAP_PROP_POS_FRAMES and value == 0:
            self.position = 0
            return True
        return False

    def release(self) -> None:
        self.released = True


def test_video_file_source_timestamps_increase_across_loop() -> None:
    capture = _Capture()
    source = VideoFileSource(
        "unused.mp4", max_loops=2, capture_factory=lambda _: capture
    )

    packets = [source.read() for _ in range(4)]

    assert [packet.timestamp_ms for packet in packets if packet] == [0, 500, 1000, 1500]
    assert source.read() is None
    source.close()
    assert capture.released is True

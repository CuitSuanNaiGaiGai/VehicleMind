from __future__ import annotations

import time
import math

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
    pixels: np.ndarray
    timestamp_ms: int
    captured_at: float


class VideoFileSource:
    """Read prerecorded frames with a monotonic timeline across video loops."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_loops: int | None = None,
        capture_factory: Callable[[str], Any] = cv2.VideoCapture,
    ) -> None:
        if max_loops is not None and (type(max_loops) is not int or max_loops <= 0):
            raise ValueError("max_loops must be a positive integer or null")
        self.path = Path(path)
        self._capture = capture_factory(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise FileNotFoundError(f"video could not be opened: {self.path}")
        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.fps = (
            reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 30.0
        )
        self.max_loops = max_loops
        self._completed_loops = 0
        self._next_frame_index = 0
        self._closed = False

    def read(self) -> VideoFrame | None:
        if self._closed:
            return None
        ok, pixels = self._capture.read()
        if not ok:
            self._completed_loops += 1
            if self.max_loops is not None and self._completed_loops >= self.max_loops:
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, pixels = self._capture.read()
            if not ok:
                return None
        packet = VideoFrame(
            pixels=pixels,
            timestamp_ms=round(self._next_frame_index * 1000 / self.fps),
            captured_at=time.monotonic(),
        )
        self._next_frame_index += 1
        return packet

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

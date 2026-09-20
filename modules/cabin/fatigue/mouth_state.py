from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MouthStateResult:
    mar: float
    is_open: bool


class MouthStateAnalyzer:
    """
    Estimate mouth opening using a normalized Mouth Aspect Ratio (MAR).

    MAR = vertical lip distance / mouth width

    Landmarks:
        13  : upper inner lip
        14  : lower inner lip
        61  : left mouth corner
        291 : right mouth corner
    """

    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    def __init__(
        self,
        mar_threshold: float = 0.35,
    ):
        self.mar_threshold = mar_threshold

    @staticmethod
    def _to_pixel(
        landmark,
        width: int,
        height: int,
    ) -> np.ndarray:

        return np.array(
            [
                landmark.x * width,
                landmark.y * height,
            ],
            dtype=np.float32,
        )

    def analyze(
        self,
        landmarks: Sequence,
        image_width: int,
        image_height: int,
    ) -> MouthStateResult:

        upper = self._to_pixel(
            landmarks[self.UPPER_LIP],
            image_width,
            image_height,
        )

        lower = self._to_pixel(
            landmarks[self.LOWER_LIP],
            image_width,
            image_height,
        )

        left = self._to_pixel(
            landmarks[self.LEFT_CORNER],
            image_width,
            image_height,
        )

        right = self._to_pixel(
            landmarks[self.RIGHT_CORNER],
            image_width,
            image_height,
        )

        vertical = np.linalg.norm(
            upper - lower
        )

        horizontal = np.linalg.norm(
            left - right
        )

        if horizontal < 1e-6:
            mar = 0.0
        else:
            mar = float(
                vertical / horizontal
            )

        return MouthStateResult(
            mar=mar,
            is_open=mar > self.mar_threshold,
        )

    @classmethod
    def landmark_indices(
        cls,
    ) -> tuple[int, ...]:

        return (
            cls.UPPER_LIP,
            cls.LOWER_LIP,
            cls.LEFT_CORNER,
            cls.RIGHT_CORNER,
        )

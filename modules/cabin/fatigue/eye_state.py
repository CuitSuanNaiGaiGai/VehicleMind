from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class EyeStateResult:
    left_ear: float
    right_ear: float
    mean_ear: float
    is_closed: bool


class EyeStateAnalyzer:
    """
    Eye-state estimation based on Eye Aspect Ratio (EAR).

    Note:
        EAR threshold is heuristic and person-dependent.
        At this stage it is only used to verify the eye-state signal,
        not to directly determine fatigue.
    """

    # MediaPipe Face Mesh landmark indices
    #
    # Point order:
    # p1: outer corner
    # p2: upper eyelid
    # p3: upper eyelid
    # p4: inner corner
    # p5: lower eyelid
    # p6: lower eyelid

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)

    def __init__(self, ear_threshold: float = 0.21):
        self.ear_threshold = ear_threshold

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

    def _compute_ear(
        self,
        landmarks: Sequence,
        indices: tuple[int, ...],
        width: int,
        height: int,
    ) -> float:
        points = [
            self._to_pixel(
                landmarks[index],
                width,
                height,
            )
            for index in indices
        ]

        p1, p2, p3, p4, p5, p6 = points

        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        if horizontal < 1e-6:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return float(ear)

    def analyze(
        self,
        landmarks: Sequence,
        image_width: int,
        image_height: int,
    ) -> EyeStateResult:
        left_ear = self._compute_ear(
            landmarks,
            self.LEFT_EYE,
            image_width,
            image_height,
        )

        right_ear = self._compute_ear(
            landmarks,
            self.RIGHT_EYE,
            image_width,
            image_height,
        )

        mean_ear = (left_ear + right_ear) / 2.0

        return EyeStateResult(
            left_ear=left_ear,
            right_ear=right_ear,
            mean_ear=mean_ear,
            is_closed=mean_ear < self.ear_threshold,
        )

    @classmethod
    def eye_indices(cls) -> tuple[int, ...]:
        return cls.LEFT_EYE + cls.RIGHT_EYE

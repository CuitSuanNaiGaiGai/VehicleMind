from dataclasses import dataclass

import numpy as np


@dataclass
class DrivingObject:
    """One detected traffic or road object."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DrivingSceneResult:
    """Unified road-perception output at the input frame resolution."""

    objects: list[DrivingObject]
    drivable_mask: np.ndarray
    lane_mask: np.ndarray
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    total_ms: float

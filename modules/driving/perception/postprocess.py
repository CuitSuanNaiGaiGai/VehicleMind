from __future__ import annotations

import cv2
import numpy as np

from modules.driving.perception import yolopv2_utils


def decode_masks(
    outputs,
    pad: tuple[float, float],
    original_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode drivable-area and lane probability maps."""
    pad_w, pad_h = pad
    drivable_prob = np.asarray(
        yolopv2_utils.driving_area_mask(outputs[4], (pad_w, pad_h)),
        dtype=np.float32,
    )
    lane_prob = np.asarray(
        yolopv2_utils.lane_line_mask(outputs[5], (pad_w, pad_h)),
        dtype=np.float32,
    )

    original_width, original_height = original_size
    output_size = (original_width, original_height)
    drivable_prob = cv2.resize(
        drivable_prob,
        output_size,
        interpolation=cv2.INTER_LINEAR,
    )
    lane_prob = cv2.resize(
        lane_prob,
        output_size,
        interpolation=cv2.INTER_LINEAR,
    )
    return (
        (drivable_prob > 0.5).astype(np.uint8),
        (lane_prob > 0.5).astype(np.uint8),
    )

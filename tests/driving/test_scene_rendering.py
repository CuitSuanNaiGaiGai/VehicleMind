from __future__ import annotations

import numpy as np

from apps.driving_demo.ui.scene import (
    draw_drivable_area,
    draw_lane_mask,
    draw_object,
)
from modules.driving.perception.types import DrivingObject


def test_drivable_area_blends_only_masked_pixels_with_original_color() -> None:
    frame = np.full((3, 3, 3), 100, dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1

    draw_drivable_area(frame, mask, alpha=0.5)

    assert frame[1, 1].tolist() == [140, 50, 95]
    assert frame[0, 0].tolist() == [100, 100, 100]


def test_lane_mask_keeps_dilated_filled_yellow_rendering() -> None:
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1

    draw_lane_mask(frame, mask)

    assert frame[2, 2].tolist() == [0, 230, 255]
    assert frame[1, 2].tolist() == [0, 230, 255]
    assert frame[0, 0].tolist() == [0, 0, 0]


def test_object_box_coordinates_are_clipped_to_frame_boundaries() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    obj = DrivingObject(
        x1=-10,
        y1=-5,
        x2=50,
        y2=45,
        confidence=0.9,
        class_id=2,
        class_name="car",
    )

    draw_object(frame, obj)

    assert np.any(frame[-1, 0] != 0)
    assert np.any(frame[-1, -1] != 0)

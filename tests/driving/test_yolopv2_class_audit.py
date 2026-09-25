from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_yolopv2_class_ids import (
    count_class_ids,
    safe_asset_reference,
    sample_frame_indices,
)


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [
        (1, [0]),
        (2, [0, 1]),
        (1071, [0, 535, 1070]),
    ],
)
def test_sample_frame_indices_selects_unique_first_middle_last(
    frame_count: int,
    expected: list[int],
) -> None:
    assert sample_frame_indices(frame_count) == expected


def test_sample_frame_indices_rejects_empty_video() -> None:
    with pytest.raises(ValueError, match="frame_count"):
        sample_frame_indices(0)


def test_count_class_ids_returns_sorted_string_keys() -> None:
    detections = [
        [0.0, 0.0, 1.0, 1.0, 0.9, 3.0],
        [0.0, 0.0, 1.0, 1.0, 0.8, 2.0],
        [0.0, 0.0, 1.0, 1.0, 0.7, 3.0],
    ]

    assert count_class_ids(detections) == {"2": 1, "3": 2}
    assert count_class_ids([]) == {}


def test_safe_asset_reference_keeps_basename_without_local_path() -> None:
    reference = safe_asset_reference(
        Path("/private/data/road_test.mp4"),
        "a" * 64,
    )

    assert reference == {"filename": "road_test.mp4", "sha256": "a" * 64}
    assert "/private/data" not in str(reference)

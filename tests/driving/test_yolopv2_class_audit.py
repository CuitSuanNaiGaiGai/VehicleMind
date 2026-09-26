from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_yolopv2_class_ids import (
    count_class_ids,
    safe_asset_reference,
    safe_input_shape,
    sample_frame_indices,
    run_non_max_suppression,
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


def test_safe_input_shape_hides_symbolic_metadata() -> None:
    assert safe_input_shape([1, 3, "/private/user/model.onnx", "width"]) == [
        1,
        3,
        "dynamic",
        "dynamic",
    ]


def test_nms_warning_does_not_pollute_json_stdout(monkeypatch, capsys) -> None:
    expected = [["detections"]]

    def noisy_nms(*args, **kwargs):
        print("WARNING: NMS time limit exceeded")
        return expected

    monkeypatch.setattr(
        "scripts.audit_yolopv2_class_ids.yolopv2_utils.non_max_suppression",
        noisy_nms,
    )

    result = run_non_max_suppression("prediction")

    captured = capsys.readouterr()
    assert result is expected
    assert captured.out == ""
    assert "WARNING: NMS time limit exceeded" in captured.err

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from modules.config.perception import PerceptionConfig


def test_repository_perception_config_loads() -> None:
    config = PerceptionConfig.load_default()

    assert config.phone.confidence_threshold == pytest.approx(0.35)
    assert config.driving.nms_threshold == pytest.approx(0.45)
    assert config.lane.canny_low == 60
    assert config.lane.white_hls_lower == (0, 160, 0)
    assert config.lane.fit_top_height == pytest.approx(0.55)


def test_partial_config_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "perception.yaml"
    path.write_text(
        "phone:\n  confidence_threshold: 0.42\ndriving:\n  work_width: 960\n",
        encoding="utf-8",
    )

    config = PerceptionConfig.load(path)

    assert config.phone.confidence_threshold == pytest.approx(0.42)
    assert config.phone.image_size == 640
    assert config.driving.work_width == 960
    assert config.driving.nms_threshold == pytest.approx(0.45)


def test_invalid_threshold_order_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "perception.yaml"
    path.write_text(
        "lane:\n  canny_low: 180\n  canny_high: 120\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lane.canny_low"):
        PerceptionConfig.load(path)


def test_invalid_roi_order_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "perception.yaml"
    path.write_text(
        "lane:\n  roi_top_left: 0.7\n  roi_top_right: 0.6\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lane.roi_top_left"):
        PerceptionConfig.load(path)


def test_invalid_hls_triplet_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "perception.yaml"
    path.write_text(
        "lane:\n  yellow_hls_lower: [10, 80, 300]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lane.yellow_hls_lower"):
        PerceptionConfig.load(path)


def test_unknown_perception_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "perception.yaml"
    path.write_text("phone:\n  magic_threshold: 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="phone.magic_threshold"):
        PerceptionConfig.load(path)


def test_perception_configuration_is_immutable() -> None:
    config = PerceptionConfig.load_default()

    with pytest.raises(FrozenInstanceError):
        config.phone.image_size = 320  # type: ignore[misc]

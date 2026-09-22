from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from apps.driving_demo.object_cli import parse_object_args
from apps.driving_demo.scene_cli import parse_scene_args
from modules.config import PerceptionConfig
from modules.config.overrides import resolve_overrides
from modules.driving.lane.lane_detector import LaneDetector


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINTS = (
    "apps/cabin_demo/phone_demo.py",
    "apps/driving_demo/scene_runner.py",
    "apps/driving_demo/object_demo.py",
    "apps/driving_demo/lane_demo.py",
)
CONFIGURED_CONSTRUCTORS = {
    "PhoneDetector",
    "PhoneBehaviorTracker",
    "PanopticDrivingDetector",
    "RoadObjectDetector",
    "LaneDetector",
}


def test_explicit_override_wins_without_mutating_defaults() -> None:
    original = PerceptionConfig.load_default()

    resolved = resolve_overrides(
        original,
        {"driving.score_threshold": 0.42, "phone.image_size": 320},
    )

    assert resolved.driving.score_threshold == pytest.approx(0.42)
    assert resolved.phone.image_size == 320
    assert original.driving.score_threshold == pytest.approx(0.30)
    assert original.phone.image_size == 640


def test_invalid_override_uses_domain_validation() -> None:
    with pytest.raises(ValueError, match="driving.nms_threshold"):
        resolve_overrides(
            PerceptionConfig.load_default(),
            {"driving.nms_threshold": 1.2},
        )


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown perception override"):
        resolve_overrides(
            PerceptionConfig.load_default(),
            {"driving.magic_threshold": 0.2},
        )


def test_scene_cli_resolves_defaults_and_explicit_overrides() -> None:
    defaults = PerceptionConfig.load_default()

    default_args = parse_scene_args([], config=defaults)
    overridden = parse_scene_args(
        ["--conf", "0.42", "--work-width", "960"],
        config=defaults,
    )

    assert default_args.conf == pytest.approx(0.30)
    assert default_args.work_width == 1280
    assert overridden.conf == pytest.approx(0.42)
    assert overridden.work_width == 960


def test_object_cli_resolves_defaults_and_explicit_overrides() -> None:
    defaults = PerceptionConfig.load_default()

    default_args = parse_object_args([], config=defaults)
    overridden = parse_object_args(
        ["--conf", "0.52", "--imgsz", "320"],
        config=defaults,
    )

    assert default_args.model == "yolo11n.pt"
    assert default_args.conf == pytest.approx(0.25)
    assert overridden.conf == pytest.approx(0.52)
    assert overridden.imgsz == 320


def test_lane_detector_uses_injected_configuration() -> None:
    lane = replace(
        PerceptionConfig.load_default().lane,
        smoothing=0.6,
        canny_low=70,
        canny_high=160,
    )

    detector = LaneDetector(config=lane)

    assert detector.config == lane
    assert detector.smoothing == pytest.approx(0.6)


def test_entry_points_do_not_pass_numeric_threshold_literals() -> None:
    offenders: list[str] = []
    for relative_path in ENTRY_POINTS:
        tree = ast.parse((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name not in CONFIGURED_CONSTRUCTORS:
                continue
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant) and isinstance(
                    keyword.value.value, (int, float)
                ):
                    offenders.append(f"{relative_path}:{node.lineno}:{keyword.arg}")

    assert offenders == []

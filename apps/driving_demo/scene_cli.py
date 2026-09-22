from __future__ import annotations

import argparse

from collections.abc import Sequence

from modules.config import PerceptionConfig
from modules.config.overrides import resolve_overrides


def parse_scene_args(
    argv: Sequence[str] | None = None,
    *,
    config: PerceptionConfig | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VehicleMind Integrated Driving Perception Demo"
    )
    parser.add_argument("--video", default="assets/driving/road_test.mp4")
    parser.add_argument("--output", default="assets/driving/scene_result.mp4")
    parser.add_argument(
        "--panoptic-model",
        default="models/driving/YOLOPv2_512.onnx",
    )
    parser.add_argument("--work-width", type=int)
    parser.add_argument("--work-height", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--nms", type=float)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    base = config or PerceptionConfig.load_default()
    overrides: dict[str, object] = {}
    for argument, field_name in (
        ("work_width", "work_width"),
        ("work_height", "work_height"),
        ("conf", "score_threshold"),
        ("nms", "nms_threshold"),
    ):
        value = getattr(args, argument)
        if value is not None:
            overrides[f"driving.{field_name}"] = value
    driving = resolve_overrides(base, overrides).driving
    args.work_width = driving.work_width
    args.work_height = driving.work_height
    args.conf = driving.score_threshold
    args.nms = driving.nms_threshold
    args.prefer_coreml = driving.prefer_coreml
    args.warmup_runs = driving.warmup_runs
    return args

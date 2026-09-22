from __future__ import annotations

import argparse

from collections.abc import Sequence

from modules.config import PerceptionConfig
from modules.config.overrides import resolve_overrides


def parse_object_args(
    argv: Sequence[str] | None = None,
    *,
    config: PerceptionConfig | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VehicleMind Road Object Detection Demo"
    )
    parser.add_argument("--video", default="assets/driving/road_test.mp4")
    parser.add_argument("--output", default="assets/driving/object_result.mp4")
    parser.add_argument("--model")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    base = config or PerceptionConfig.load_default()
    overrides: dict[str, object] = {}
    if args.model is not None:
        overrides["driving.object_model_name"] = args.model
    if args.conf is not None:
        overrides["driving.object_confidence_threshold"] = args.conf
    if args.imgsz is not None:
        overrides["driving.object_image_size"] = args.imgsz
    driving = resolve_overrides(base, overrides).driving
    args.model = driving.object_model_name
    args.conf = driving.object_confidence_threshold
    args.imgsz = driving.object_image_size
    return args

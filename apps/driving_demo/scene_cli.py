from __future__ import annotations

import argparse


def parse_scene_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VehicleMind Integrated Driving Perception Demo"
    )
    parser.add_argument("--video", default="assets/driving/road_test.mp4")
    parser.add_argument("--output", default="assets/driving/scene_result.mp4")
    parser.add_argument(
        "--panoptic-model",
        default="models/driving/YOLOPv2_512.onnx",
    )
    parser.add_argument("--work-width", type=int, default=1280)
    parser.add_argument("--work-height", type=int, default=720)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

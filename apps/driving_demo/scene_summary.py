from __future__ import annotations

from pathlib import Path


def print_performance_summary(
    *,
    frame_count: int,
    source_fps: float,
    total_elapsed: float,
    output_path: Path,
    preprocess_ms: float,
    inference_ms: float,
    postprocess_ms: float,
    detector_ms: float,
    pipeline_ms: float,
) -> None:
    divisor = max(frame_count, 1)
    averages = {
        "Preprocess": preprocess_ms / divisor,
        "Inference": inference_ms / divisor,
        "Postprocess": postprocess_ms / divisor,
        "Detector total": detector_ms / divisor,
        "Whole pipeline": pipeline_ms / divisor,
    }
    pipeline_fps = 1000.0 / max(averages["Whole pipeline"], 1e-6)
    frame_budget = 1000.0 / source_fps

    print("\n========================================")
    print(" VehicleMind Performance Summary")
    print("========================================")
    print(f"Processed frames : {frame_count}")
    print(f"Source FPS       : {source_fps:.2f}")
    print(f"Frame budget     : {frame_budget:.2f} ms\n")
    for label, value in averages.items():
        print(f"{label:<17}: {value:.2f} ms")
    print(f"Pipeline FPS     : {pipeline_fps:.2f}\n")
    status = "READY" if averages["Whole pipeline"] <= frame_budget else "NOT YET"
    print(f"Realtime status  : {status}")
    print(f"Wall-clock time  : {total_elapsed:.2f} s")
    print(f"Output           : {output_path}")
    print("========================================")

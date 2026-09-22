from __future__ import annotations

import time
from pathlib import Path

import cv2

from apps.driving_demo.scene_cli import parse_scene_args
from apps.driving_demo.scene_summary import print_performance_summary
from apps.driving_demo.ui.scene import (
    draw_dashboard,
    draw_drivable_area,
    draw_lane_mask,
    draw_object,
)
from modules.driving.perception.panoptic_detector import PanopticDrivingDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _open_writer(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {path}")
    return writer


def run_scene_demo() -> None:
    args = parse_scene_args()
    input_path = PROJECT_ROOT / args.video
    output_path = PROJECT_ROOT / args.output
    model_path = PROJECT_ROOT / args.panoptic_model
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"YOLOPv2 model not found: {model_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = PanopticDrivingDetector(
        model_path=model_path,
        score_threshold=args.conf,
        nms_threshold=args.nms,
        prefer_coreml=True,
        warmup_runs=2,
    )
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    work_size = (int(args.work_width), int(args.work_height))
    writer = _open_writer(output_path, source_fps, work_size)
    print(f"[VehicleMind] Input: {input_path}")
    print(f"[VehicleMind] Working resolution: {work_size[0]}x{work_size[1]}")
    print(f"[VehicleMind] Source FPS: {source_fps:.2f}")

    frame_count = 0
    pipeline_fps = 0.0
    totals = {
        "preprocess": 0.0,
        "inference": 0.0,
        "postprocess": 0.0,
        "detector": 0.0,
        "pipeline": 0.0,
    }
    started_at = time.perf_counter()
    try:
        while True:
            loop_started_at = time.perf_counter()
            success, source_frame = capture.read()
            if not success:
                break

            frame_count += 1
            frame = cv2.resize(source_frame, work_size, interpolation=cv2.INTER_AREA)
            result = detector.detect(frame)
            draw_drivable_area(frame, result.drivable_mask)
            draw_lane_mask(frame, result.lane_mask)
            for obj in result.objects:
                draw_object(frame, obj)

            elapsed = time.perf_counter() - loop_started_at
            current_fps = 1.0 / max(elapsed, 1e-6)
            pipeline_fps = (
                current_fps
                if pipeline_fps <= 0
                else (0.9 * pipeline_fps + 0.1 * current_fps)
            )
            draw_dashboard(frame, result, pipeline_fps, source_fps)
            writer.write(frame)

            if args.show:
                cv2.imshow("VehicleMind - Driving Perception", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            loop_ms = (time.perf_counter() - loop_started_at) * 1000.0
            totals["preprocess"] += result.preprocess_ms
            totals["inference"] += result.inference_ms
            totals["postprocess"] += result.postprocess_ms
            totals["detector"] += result.total_ms
            totals["pipeline"] += loop_ms
            if frame_count % 50 == 0:
                progress = frame_count / total_frames * 100 if total_frames else 0.0
                print(
                    f"[VehicleMind] {frame_count}/{total_frames} "
                    f"({progress:.1f}%) | {pipeline_fps:.1f} FPS"
                )
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print_performance_summary(
        frame_count=frame_count,
        source_fps=source_fps,
        total_elapsed=time.perf_counter() - started_at,
        output_path=output_path,
        preprocess_ms=totals["preprocess"],
        inference_ms=totals["inference"],
        postprocess_ms=totals["postprocess"],
        detector_ms=totals["detector"],
        pipeline_ms=totals["pipeline"],
    )

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from modules.driving.perception.panoptic_detector import (
    DrivingObject,
    DrivingSceneResult,
)


OBJECT_COLORS = {
    "person": (0, 80, 255),
    "rider": (255, 160, 0),
    "car": (80, 220, 80),
    "truck": (255, 120, 80),
    "bus": (0, 200, 255),
    "train": (220, 120, 255),
    "motorcycle": (255, 100, 255),
    "bicycle": (255, 180, 0),
    "traffic light": (0, 255, 255),
    "traffic sign": (0, 0, 255),
}


def _text(
    frame,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (225, 225, 225),
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_drivable_area(
    frame: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.22,
) -> None:
    overlay = np.zeros_like(frame)
    overlay[mask > 0] = (40, 110, 40)
    cv2.addWeighted(overlay, alpha, frame, 1.0, 0, frame)


def draw_lane_mask(frame: np.ndarray, mask: np.ndarray) -> None:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)


def draw_object(frame: np.ndarray, obj: DrivingObject) -> None:
    color = OBJECT_COLORS.get(obj.class_name, (200, 200, 200))
    cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), color, 2)
    label = f"{obj.class_name.upper()} {obj.confidence:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1,
    )
    label_top = max(0, obj.y1 - label_height - 10)
    cv2.rectangle(
        frame,
        (obj.x1, label_top),
        (obj.x1 + label_width + 8, obj.y1),
        color,
        -1,
    )
    _text(
        frame,
        label,
        (obj.x1 + 4, obj.y1 - 5),
        scale=0.48,
        color=(20, 20, 20),
        thickness=1,
    )


def draw_dashboard(
    frame: np.ndarray,
    scene_result: DrivingSceneResult,
    pipeline_fps: float,
    source_fps: float,
) -> None:
    height, width = frame.shape[:2]
    panel_width = 335
    left = max(0, width - panel_width)
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, 0), (width, height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    x, y = left + 22, 38
    _text(frame, "VEHICLEMIND", (x, y), scale=0.78, thickness=2)
    y += 27
    _text(frame, "DRIVING PERCEPTION", (x, y), color=(170, 170, 170))
    y += 40

    counts = Counter(obj.class_name for obj in scene_result.objects)
    rows = [
        ("Objects", str(len(scene_result.objects))),
        ("Vehicles", str(sum(counts[name] for name in ("car", "truck", "bus")))),
        ("People", str(counts["person"] + counts["rider"])),
        ("Drivable area", "DETECTED" if np.any(scene_result.drivable_mask) else "NONE"),
        ("Lane marks", "DETECTED" if np.any(scene_result.lane_mask) else "NONE"),
    ]
    for label, value in rows:
        _text(frame, label, (x, y))
        _text(frame, value, (x + 165, y), color=(80, 220, 80), thickness=2)
        y += 32

    y += 15
    cv2.line(frame, (x, y), (width - 20, y), (90, 90, 90), 1)
    y += 32
    timing_rows = [
        ("Preprocess", scene_result.preprocess_ms),
        ("Inference", scene_result.inference_ms),
        ("Postprocess", scene_result.postprocess_ms),
        ("Detector", scene_result.total_ms),
    ]
    for label, value in timing_rows:
        _text(frame, label, (x, y))
        _text(frame, f"{value:.1f} ms", (x + 165, y))
        y += 30

    realtime = pipeline_fps >= source_fps
    y += 15
    _text(frame, f"Pipeline {pipeline_fps:.1f} FPS", (x, y), thickness=2)
    y += 30
    _text(
        frame,
        "REALTIME READY" if realtime else "BELOW SOURCE FPS",
        (x, y),
        color=(80, 220, 80) if realtime else (0, 165, 255),
        thickness=2,
    )

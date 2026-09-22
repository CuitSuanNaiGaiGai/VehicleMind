import time
from pathlib import Path

import cv2

from apps.driving_demo.object_cli import parse_object_args
from apps.driving_demo.ui.objects import draw_dashboard, draw_object
from modules.driving.detection.object_detector import (
    RoadObjectDetector,
)


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Colors
# ============================================================

OBJECT_COLORS = {
    "person": (0, 80, 255),
    "bicycle": (255, 180, 0),
    "car": (80, 220, 80),
    "motorcycle": (255, 100, 255),
    "bus": (0, 200, 255),
    "truck": (255, 120, 80),
    "traffic light": (0, 255, 255),
    "stop sign": (0, 0, 255),
}


def main():
    args = parse_object_args()

    input_path = PROJECT_ROOT / args.video

    output_path = PROJECT_ROOT / args.output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    # ========================================================
    # Detector
    # ========================================================

    detector = RoadObjectDetector(
        model_name=args.model,
        confidence_threshold=(args.conf),
        image_size=(args.imgsz),
    )

    # ========================================================
    # Input video
    # ========================================================

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if source_fps <= 0:
        source_fps = 25.0

    print(f"[VehicleMind] Input: {input_path}")

    print(f"[VehicleMind] Resolution: {width}x{height}")

    print(f"[VehicleMind] FPS: {source_fps:.2f}")

    print(f"[VehicleMind] Frames: {total_frames}")

    # ========================================================
    # Output video
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        source_fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(f"Failed to create output video: {output_path}")

    # ========================================================
    # Runtime statistics
    # ========================================================

    frame_index = 0

    fps = 0.0

    fps_counter = 0

    fps_start = time.perf_counter()

    # ========================================================
    # Processing loop
    # ========================================================

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_index += 1

            # =================================================
            # Object Detection
            # =================================================

            result = detector.detect(frame)

            # =================================================
            # Draw objects
            # =================================================

            for obj in result.objects:
                draw_object(
                    frame,
                    obj,
                )

            # =================================================
            # Processing FPS
            # =================================================

            fps_counter += 1

            elapsed = time.perf_counter() - fps_start

            if elapsed >= 1.0:
                fps = fps_counter / elapsed

                fps_counter = 0

                fps_start = time.perf_counter()

            # =================================================
            # Dashboard
            # =================================================

            draw_dashboard(
                frame,
                result.counts,
                fps,
            )

            # =================================================
            # Frame progress
            # =================================================

            if frame_index % 100 == 0:
                if total_frames > 0:
                    progress = frame_index / total_frames * 100.0

                    print(
                        f"[VehicleMind] {frame_index}/{total_frames} ({progress:.1f}%)"
                    )

            # =================================================
            # Save
            # =================================================

            writer.write(frame)

            # =================================================
            # Preview
            # =================================================

            if args.show:
                preview = frame

                # Avoid huge 4K preview windows.
                if width > 1600:
                    preview_width = 1280

                    preview_height = int(height * preview_width / width)

                    preview = cv2.resize(
                        frame,
                        (
                            preview_width,
                            preview_height,
                        ),
                    )

                cv2.imshow(
                    ("VehicleMind - Driving Perception"),
                    preview,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    finally:
        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    print()

    print("[VehicleMind] Object detection finished.")

    print(f"[VehicleMind] Output: {output_path}")


if __name__ == "__main__":
    main()

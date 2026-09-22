import argparse
import time
from pathlib import Path

import cv2

from apps.driving_demo.ui.lane import draw_dashboard, draw_lane_result
from modules.config import PerceptionConfig
from modules.driving.lane.lane_detector import (
    LaneDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():

    parser = argparse.ArgumentParser(description=("VehicleMind Lane Perception Demo"))

    parser.add_argument(
        "--video",
        type=str,
        default=("assets/driving/road_test.mp4"),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=("assets/driving/lane_result.mp4"),
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    args = parser.parse_args()

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

    detector = LaneDetector(config=PerceptionConfig.load_default().lane)

    # ========================================================
    # Video
    # ========================================================

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0:
        source_fps = 25.0

    print(f"[VehicleMind] Input: {input_path}")

    print(f"[VehicleMind] Resolution: {width}x{height}")

    print(f"[VehicleMind] Frames: {total_frames}")

    # ========================================================
    # Writer
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

        raise RuntimeError(f"Failed to create {output_path}")

    # ========================================================
    # FPS
    # ========================================================

    fps = 0.0
    fps_counter = 0

    fps_start = time.perf_counter()

    frame_index = 0

    # ========================================================
    # Processing
    # ========================================================

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_index += 1

            # =================================================
            # Lane perception
            # =================================================

            result = detector.detect(frame)

            # =================================================
            # Runtime FPS
            # =================================================

            fps_counter += 1

            elapsed = time.perf_counter() - fps_start

            if elapsed >= 1.0:
                fps = fps_counter / elapsed

                fps_counter = 0

                fps_start = time.perf_counter()

            # =================================================
            # Visualization
            # =================================================

            draw_lane_result(
                frame,
                result,
            )

            draw_dashboard(
                frame,
                result,
                fps,
            )

            # =================================================
            # Save
            # =================================================

            writer.write(frame)

            # =================================================
            # Progress
            # =================================================

            if frame_index % 100 == 0:
                if total_frames > 0:
                    progress = frame_index / total_frames * 100.0

                    print(
                        f"[VehicleMind] {frame_index}/{total_frames} ({progress:.1f}%)"
                    )

            # =================================================
            # Preview
            # =================================================

            if args.show:
                preview = frame

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
                    ("VehicleMind - Lane Perception"),
                    preview,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    finally:
        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    print("[VehicleMind] Lane perception finished.")

    print(f"[VehicleMind] Output: {output_path}")


if __name__ == "__main__":
    main()

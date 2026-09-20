import argparse
import time
from pathlib import Path

import cv2

from modules.driving.detection.object_detector import (
    RoadObjectDetector,
)


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


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


# ============================================================
# Drawing utilities
# ============================================================

def draw_object(
    frame,
    obj,
) -> None:

    color = OBJECT_COLORS.get(
        obj.class_name,
        (220, 220, 220),
    )

    cv2.rectangle(
        frame,
        (
            obj.x1,
            obj.y1,
        ),
        (
            obj.x2,
            obj.y2,
        ),
        color,
        2,
    )

    label = (
        f"{obj.class_name.upper()} "
        f"{obj.confidence:.2f}"
    )

    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    (
        text_width,
        text_height,
    ), baseline = (
        cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
    )

    label_y1 = max(
        0,
        obj.y1
        - text_height
        - baseline
        - 8,
    )

    cv2.rectangle(
        frame,
        (
            obj.x1,
            label_y1,
        ),
        (
            obj.x1
            + text_width
            + 10,
            obj.y1,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            obj.x1 + 5,
            obj.y1 - 6,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )


def draw_dashboard(
    frame,
    counts,
    fps: float,
) -> None:

    height, width = (
        frame.shape[:2]
    )

    panel_width = 300

    x1 = (
        width
        - panel_width
    )

    # --------------------------------------------------------
    # Transparent panel
    # --------------------------------------------------------

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,
        (
            x1,
            0,
        ),
        (
            width,
            height,
        ),
        (
            18,
            18,
            18,
        ),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame,
    )

    left = (
        x1 + 22
    )

    y = 42

    # ========================================================
    # Header
    # ========================================================

    cv2.putText(
        frame,
        "VEHICLEMIND",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y += 30

    cv2.putText(
        frame,
        "DRIVING PERCEPTION",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    y += 25

    cv2.line(
        frame,
        (
            left,
            y,
        ),
        (
            width - 20,
            y,
        ),
        (
            90,
            90,
            90,
        ),
        1,
    )

    y += 35

    cv2.putText(
        frame,
        "ROAD OBJECTS",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (170, 170, 170),
        2,
        cv2.LINE_AA,
    )

    y += 35

    # ========================================================
    # Important categories
    # ========================================================

    display_classes = (
        "car",
        "truck",
        "bus",
        "person",
        "bicycle",
        "motorcycle",
        "traffic light",
        "stop sign",
    )

    for class_name in (
        display_classes
    ):

        count = counts.get(
            class_name,
            0,
        )

        label = (
            class_name
            .replace(
                "_",
                " ",
            )
            .title()
        )

        cv2.putText(
            frame,
            label,
            (
                left,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )

        color = (
            OBJECT_COLORS.get(
                class_name,
                (
                    220,
                    220,
                    220,
                ),
            )
        )

        cv2.putText(
            frame,
            str(count),
            (
                left + 190,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

        y += 31

    y += 15

    cv2.line(
        frame,
        (
            left,
            y,
        ),
        (
            width - 20,
            y,
        ),
        (
            90,
            90,
            90,
        ),
        1,
    )

    y += 35

    # ========================================================
    # System
    # ========================================================

    total_objects = sum(
        counts.values()
    )

    cv2.putText(
        frame,
        "SYSTEM",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (170, 170, 170),
        2,
        cv2.LINE_AA,
    )

    y += 35

    cv2.putText(
        frame,
        (
            f"Objects    "
            f"{total_objects}"
        ),
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )

    y += 30

    cv2.putText(
        frame,
        (
            f"FPS        "
            f"{fps:.1f}"
        ),
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "VehicleMind "
                "Road Object Detection Demo"
            )
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=(
            "assets/driving/"
            "road_test.mp4"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "assets/driving/"
            "object_result.mp4"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    input_path = (
        PROJECT_ROOT
        / args.video
    )

    output_path = (
        PROJECT_ROOT
        / args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Video not found: "
            f"{input_path}"
        )

    # ========================================================
    # Detector
    # ========================================================

    detector = (
        RoadObjectDetector(
            model_name=args.model,
            confidence_threshold=(
                args.conf
            ),
            image_size=(
                args.imgsz
            ),
        )
    )

    # ========================================================
    # Input video
    # ========================================================

    cap = (
        cv2.VideoCapture(
            str(
                input_path
            )
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Failed to open video: "
            f"{input_path}"
        )

    source_fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if source_fps <= 0:

        source_fps = 25.0

    print(
        "[VehicleMind] "
        f"Input: {input_path}"
    )

    print(
        "[VehicleMind] "
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        "[VehicleMind] "
        f"FPS: {source_fps:.2f}"
    )

    print(
        "[VehicleMind] "
        f"Frames: {total_frames}"
    )

    # ========================================================
    # Output video
    # ========================================================

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = (
        cv2.VideoWriter(
            str(
                output_path
            ),
            fourcc,
            source_fps,
            (
                width,
                height,
            ),
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Failed to create output "
            f"video: {output_path}"
        )

    # ========================================================
    # Runtime statistics
    # ========================================================

    frame_index = 0

    fps = 0.0

    fps_counter = 0

    fps_start = (
        time.perf_counter()
    )

    # ========================================================
    # Processing loop
    # ========================================================

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                break

            frame_index += 1

            # =================================================
            # Object Detection
            # =================================================

            result = (
                detector.detect(
                    frame
                )
            )

            # =================================================
            # Draw objects
            # =================================================

            for obj in (
                result.objects
            ):

                draw_object(
                    frame,
                    obj,
                )

            # =================================================
            # Processing FPS
            # =================================================

            fps_counter += 1

            elapsed = (
                time.perf_counter()
                - fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    fps_counter
                    / elapsed
                )

                fps_counter = 0

                fps_start = (
                    time.perf_counter()
                )

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

            if (
                frame_index % 100
                == 0
            ):

                if (
                    total_frames > 0
                ):

                    progress = (
                        frame_index
                        / total_frames
                        * 100.0
                    )

                    print(
                        "[VehicleMind] "
                        f"{frame_index}/"
                        f"{total_frames} "
                        f"({progress:.1f}%)"
                    )

            # =================================================
            # Save
            # =================================================

            writer.write(
                frame
            )

            # =================================================
            # Preview
            # =================================================

            if args.show:

                preview = frame

                # Avoid huge 4K preview windows.
                if width > 1600:

                    preview_width = (
                        1280
                    )

                    preview_height = int(
                        height
                        * preview_width
                        / width
                    )

                    preview = (
                        cv2.resize(
                            frame,
                            (
                                preview_width,
                                preview_height,
                            ),
                        )
                    )

                cv2.imshow(
                    (
                        "VehicleMind - "
                        "Driving Perception"
                    ),
                    preview,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if (
                    key
                    == ord("q")
                ):
                    break

    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    print()

    print(
        "[VehicleMind] "
        "Object detection finished."
    )

    print(
        "[VehicleMind] "
        f"Output: {output_path}"
    )


if __name__ == "__main__":

    main()

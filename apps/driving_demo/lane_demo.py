import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from modules.driving.lane.lane_detector import (
    LaneDetector,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# Drawing
# ============================================================

def draw_lane_result(
    frame,
    result,
):
    height, width = (
        frame.shape[:2]
    )

    overlay = (
        frame.copy()
    )

    # ========================================================
    # Lane corridor
    # ========================================================

    if (
        result.left_lane is not None
        and
        result.right_lane is not None
    ):

        (
            lx1,
            ly1,
            lx2,
            ly2,
        ) = result.left_lane

        (
            rx1,
            ry1,
            rx2,
            ry2,
        ) = result.right_lane

        polygon = np.array(
            [
                [
                    (
                        lx1,
                        ly1,
                    ),
                    (
                        lx2,
                        ly2,
                    ),
                    (
                        rx2,
                        ry2,
                    ),
                    (
                        rx1,
                        ry1,
                    ),
                ]
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(
            overlay,
            polygon,
            (
                60,
                160,
                60,
            ),
        )

        cv2.addWeighted(
            overlay,
            0.20,
            frame,
            0.80,
            0,
            frame,
        )

    # ========================================================
    # Left lane
    # ========================================================

    if (
        result.left_lane
        is not None
    ):

        (
            x1,
            y1,
            x2,
            y2,
        ) = result.left_lane

        cv2.line(
            frame,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                255,
            ),
            7,
            cv2.LINE_AA,
        )

    # ========================================================
    # Right lane
    # ========================================================

    if (
        result.right_lane
        is not None
    ):

        (
            x1,
            y1,
            x2,
            y2,
        ) = result.right_lane

        cv2.line(
            frame,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                255,
            ),
            7,
            cv2.LINE_AA,
        )

    # ========================================================
    # Lane center
    # ========================================================

    if (
        result.lane_center
        is not None
    ):

        cv2.circle(
            frame,
            (
                result.lane_center,
                height - 30,
            ),
            7,
            (
                255,
                255,
                255,
            ),
            -1,
        )


def draw_dashboard(
    frame,
    result,
    fps,
):
    height, width = (
        frame.shape[:2]
    )

    panel_width = 310

    x1 = (
        width
        - panel_width
    )

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

    y = 45

    cv2.putText(
        frame,
        "VEHICLEMIND",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    y += 30

    cv2.putText(
        frame,
        "LANE PERCEPTION",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            170,
            170,
            170,
        ),
        1,
        cv2.LINE_AA,
    )

    y += 35

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

    y += 40

    # ========================================================
    # Status
    # ========================================================

    statuses = (
        (
            "Left Lane",
            result.left_detected,
        ),
        (
            "Right Lane",
            result.right_detected,
        ),
    )

    for (
        label,
        detected,
    ) in statuses:

        cv2.putText(
            frame,
            label,
            (
                left,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                220,
                220,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        if detected:

            text = (
                "DETECTED"
            )

            color = (
                80,
                220,
                80,
            )

        else:

            text = (
                "SEARCHING"
            )

            color = (
                0,
                180,
                255,
            )

        cv2.putText(
            frame,
            text,
            (
                left + 145,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

        y += 38

    y += 15

    # ========================================================
    # Lane corridor
    # ========================================================

    corridor_detected = (
        result.left_lane
        is not None
        and
        result.right_lane
        is not None
    )

    cv2.putText(
        frame,
        "Lane Corridor",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            220,
            220,
            220,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            "READY"
            if corridor_detected
            else "PARTIAL"
        ),
        (
            left + 145,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            (80, 220, 80)
            if corridor_detected
            else (0, 180, 255)
        ),
        2,
        cv2.LINE_AA,
    )

    y += 55

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

    y += 40

    cv2.putText(
        frame,
        "SYSTEM",
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            170,
            170,
            170,
        ),
        2,
        cv2.LINE_AA,
    )

    y += 38

    cv2.putText(
        frame,
        (
            f"FPS        {fps:.1f}"
        ),
        (
            left,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            220,
            220,
            220,
        ),
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
                "Lane Perception Demo"
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
            "lane_result.mp4"
        ),
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
        LaneDetector(
            smoothing=0.75,
        )
    )

    # ========================================================
    # Video
    # ========================================================

    cap = cv2.VideoCapture(
        str(
            input_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Failed to open: "
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
        f"Frames: {total_frames}"
    )

    # ========================================================
    # Writer
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
            "Failed to create "
            f"{output_path}"
        )

    # ========================================================
    # FPS
    # ========================================================

    fps = 0.0
    fps_counter = 0

    fps_start = (
        time.perf_counter()
    )

    frame_index = 0

    # ========================================================
    # Processing
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
            # Lane perception
            # =================================================

            result = (
                detector.detect(
                    frame
                )
            )

            # =================================================
            # Runtime FPS
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

            writer.write(
                frame
            )

            # =================================================
            # Progress
            # =================================================

            if (
                frame_index % 100
                == 0
            ):

                if total_frames > 0:

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
            # Preview
            # =================================================

            if args.show:

                preview = (
                    frame
                )

                if width > 1600:

                    preview_width = (
                        1280
                    )

                    preview_height = int(
                        height
                        * preview_width
                        / width
                    )

                    preview = cv2.resize(
                        frame,
                        (
                            preview_width,
                            preview_height,
                        ),
                    )

                cv2.imshow(
                    (
                        "VehicleMind - "
                        "Lane Perception"
                    ),
                    preview,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):
                    break

    finally:

        cap.release()

        writer.release()

        cv2.destroyAllWindows()

    print(
        "[VehicleMind] "
        "Lane perception finished."
    )

    print(
        "[VehicleMind] "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()

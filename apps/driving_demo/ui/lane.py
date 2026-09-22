from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Drawing
# ============================================================


def draw_lane_result(
    frame,
    result,
):
    height, width = frame.shape[:2]

    overlay = frame.copy()

    # ========================================================
    # Lane corridor
    # ========================================================

    if result.left_lane is not None and result.right_lane is not None:
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

    if result.left_lane is not None:
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

    if result.right_lane is not None:
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

    if result.lane_center is not None:
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
    height, width = frame.shape[:2]

    panel_width = 310

    x1 = width - panel_width

    overlay = frame.copy()

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

    left = x1 + 22

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
            text = "DETECTED"

            color = (
                80,
                220,
                80,
            )

        else:
            text = "SEARCHING"

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

    corridor_detected = result.left_lane is not None and result.right_lane is not None

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
        ("READY" if corridor_detected else "PARTIAL"),
        (
            left + 145,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        ((80, 220, 80) if corridor_detected else (0, 180, 255)),
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
        (f"FPS        {fps:.1f}"),
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

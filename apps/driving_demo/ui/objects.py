from pathlib import Path

import cv2


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

    label = f"{obj.class_name.upper()} {obj.confidence:.2f}"

    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    (
        (
            text_width,
            text_height,
        ),
        baseline,
    ) = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )

    label_y1 = max(
        0,
        obj.y1 - text_height - baseline - 8,
    )

    cv2.rectangle(
        frame,
        (
            obj.x1,
            label_y1,
        ),
        (
            obj.x1 + text_width + 10,
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

    height, width = frame.shape[:2]

    panel_width = 300

    x1 = width - panel_width

    # --------------------------------------------------------
    # Transparent panel
    # --------------------------------------------------------

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

    for class_name in display_classes:
        count = counts.get(
            class_name,
            0,
        )

        label = class_name.replace(
            "_",
            " ",
        ).title()

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

        color = OBJECT_COLORS.get(
            class_name,
            (
                220,
                220,
                220,
            ),
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

    total_objects = sum(counts.values())

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
        (f"Objects    {total_objects}"),
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
        (f"FPS        {fps:.1f}"),
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

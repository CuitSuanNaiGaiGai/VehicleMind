import cv2

from modules.cabin.fatigue.eye_state import EyeStateAnalyzer


def draw_eye_landmarks(
    frame,
    face,
) -> None:
    """
    Draw only eye landmarks used by EAR.

    Full MediaPipe face mesh is intentionally hidden
    in demo mode to keep the presentation clean.
    """

    height, width = frame.shape[:2]

    for index in EyeStateAnalyzer.eye_indices():
        landmark = face[index]

        x = int(landmark.x * width)

        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            2,
            (0, 220, 255),
            -1,
        )


def draw_demo_header(
    frame,
    fps,
) -> None:
    """
    Draw VehicleMind branding and runtime status.
    """

    cv2.putText(
        frame,
        "VEHICLEMIND",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "CABIN INTELLIGENCE",
        (25, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (26, 91),
        5,
        (0, 220, 0),
        -1,
    )

    cv2.putText(
        frame,
        "DRIVER MONITORING ACTIVE",
        (40, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (25, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

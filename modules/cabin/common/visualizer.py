import cv2
import numpy as np

from modules.cabin.state.driver_state import (
    DriverState,
    DriverStateResult,
)

from modules.cabin.assistance.rest_advisor import (
    RestStopRecommendation,
)


def _draw_text(
    frame,
    text,
    x,
    y,
    scale=0.6,
    color=(255, 255, 255),
    thickness=1,
):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_driver_status_panel(
    frame: np.ndarray,
    result: DriverStateResult,
    recommendation: RestStopRecommendation | None = None,
) -> np.ndarray:
    """
    Draw VehicleMind driver-state and safety-assistance panel.
    """

    height, width = frame.shape[:2]

    panel_width = min(380, int(width * 0.36))

    x1 = width - panel_width
    x2 = width

    overlay = frame.copy()

    #
    # Dark translucent panel
    #

    cv2.rectangle(
        overlay,
        (x1, 0),
        (x2, height),
        (20, 20, 20),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame,
    )

    x = x1 + 25

    #
    # Header
    #

    _draw_text(
        frame,
        "VEHICLEMIND",
        x,
        40,
        scale=0.75,
        color=(255, 255, 255),
        thickness=2,
    )

    _draw_text(
        frame,
        "CABIN INTELLIGENCE",
        x,
        67,
        scale=0.45,
        color=(180, 180, 180),
    )

    cv2.line(
        frame,
        (x, 85),
        (x2 - 25, 85),
        (100, 100, 100),
        1,
    )

    #
    # Driver state
    #

    _draw_text(
        frame,
        "DRIVER STATE",
        x,
        120,
        scale=0.5,
        color=(180, 180, 180),
    )

    state = result.state.value

    if result.state == DriverState.NORMAL:

        state_color = (0, 220, 0)

    elif result.state == DriverState.SUSPECTED:

        state_color = (0, 200, 255)

    elif result.state == DriverState.DROWSY:

        state_color = (0, 0, 255)

    else:

        state_color = (180, 180, 180)

    _draw_text(
        frame,
        state,
        x,
        158,
        scale=1.0,
        color=state_color,
        thickness=2,
    )

    #
    # Risk information
    #

    _draw_text(
        frame,
        f"Risk Level: {result.risk_level.value}",
        x,
        205,
        scale=0.58,
    )

    _draw_text(
        frame,
        f"PERCLOS: {result.perclos * 100:.1f}%",
        x,
        238,
        scale=0.58,
    )

    _draw_text(
        frame,
        (
            "Eye Closure: "
            f"{result.continuous_eye_closure:.1f}s"
        ),
        x,
        271,
        scale=0.58,
    )

    #
    # Explanation
    #

    cv2.line(
        frame,
        (x, 295),
        (x2 - 25, 295),
        (100, 100, 100),
        1,
    )

    _draw_text(
        frame,
        "SAFETY ASSISTANT",
        x,
        330,
        scale=0.5,
        color=(180, 180, 180),
    )

    _draw_text(
        frame,
        result.reason,
        x,
        365,
        scale=0.48,
    )

    #
    # Drowsiness assistance
    #

    if result.state == DriverState.DROWSY:

        _draw_text(
            frame,
            "Fatigue risk detected.",
            x,
            410,
            scale=0.63,
            color=(0, 0, 255),
            thickness=2,
        )

        _draw_text(
            frame,
            "A short rest is recommended.",
            x,
            443,
            scale=0.50,
        )

        if recommendation is not None:

            cv2.line(
                frame,
                (x, 470),
                (x2 - 25, 470),
                (100, 100, 100),
                1,
            )

            _draw_text(
                frame,
                "RECOMMENDED STOP",
                x,
                505,
                scale=0.5,
                color=(180, 180, 180),
            )

            _draw_text(
                frame,
                recommendation.name,
                x,
                540,
                scale=0.62,
                thickness=2,
            )

            _draw_text(
                frame,
                (
                    f"{recommendation.distance_km:.1f} km  |  "
                    f"ETA {recommendation.eta_minutes} min"
                ),
                x,
                575,
                scale=0.55,
            )

            #
            # Fake navigation button
            #

            button_y1 = 605
            button_y2 = min(
                655,
                height - 20,
            )

            if button_y2 > button_y1:

                cv2.rectangle(
                    frame,
                    (x, button_y1),
                    (x2 - 25, button_y2),
                    (0, 120, 255),
                    -1,
                )

                _draw_text(
                    frame,
                    "NAVIGATE",
                    x + 85,
                    button_y1 + 34,
                    scale=0.65,
                    thickness=2,
                )

    elif result.state == DriverState.SUSPECTED:

        _draw_text(
            frame,
            "Please stay alert.",
            x,
            410,
            scale=0.60,
            color=(0, 200, 255),
            thickness=2,
        )

    elif result.state == DriverState.NORMAL:

        _draw_text(
            frame,
            "Driver condition normal.",
            x,
            410,
            scale=0.55,
            color=(0, 220, 0),
        )

    return frame


import time
from pathlib import Path
from typing import Optional

import cv2

from modules.cabin.face.landmarks import (
    FaceLandmarkDetector,
)

from modules.cabin.fatigue.eye_state import (
    EyeStateAnalyzer,
)

from modules.cabin.fatigue.blink import (
    BlinkDetector,
)

from modules.cabin.fatigue.perclos import (
    PerclosEstimator,
)

from modules.cabin.fatigue.mouth_state import (
    MouthStateAnalyzer,
)

from modules.cabin.fatigue.yawn import (
    YawnDetector,
)

from modules.cabin.presence.driver_presence import (
    DriverPresence,
    DriverPresenceTracker,
)

from modules.cabin.state.driver_state import (
    DriverState,
    DriverStateEstimator,
)

from modules.cabin.assistance.rest_advisor import (
    MockRestAdvisor,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


# ============================================================
# Camera
# ============================================================

def open_camera(
    camera_id: int = 0,
) -> cv2.VideoCapture:
    """
    Open local camera.

    AVFoundation is preferred on macOS.
    """

    cap = cv2.VideoCapture(
        camera_id,
        cv2.CAP_AVFOUNDATION,
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(
            camera_id
        )

    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open camera. "
            "Please check camera permission."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    return cap


# ============================================================
# Landmark visualization
# ============================================================

def draw_face_landmarks(
    frame,
    face,
) -> None:
    """
    Draw lightweight facial landmarks.

    The full MediaPipe mesh contains many points, therefore
    use small markers to keep the Cabin Demo clean.
    """

    height, width = (
        frame.shape[:2]
    )

    for landmark in face:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        if (
            0 <= x < width
            and
            0 <= y < height
        ):
            cv2.circle(
                frame,
                (x, y),
                1,
                (80, 220, 120),
                -1,
            )


def draw_eye_landmarks(
    frame,
    face,
) -> None:
    """
    Highlight landmarks used for EAR.
    """

    height, width = (
        frame.shape[:2]
    )

    for index in (
        EyeStateAnalyzer.eye_indices()
    ):

        landmark = face[index]

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 255),
            -1,
        )


def draw_mouth_landmarks(
    frame,
    face,
) -> None:
    """
    Highlight landmarks used for MAR / Yawn.
    """

    height, width = (
        frame.shape[:2]
    )

    for index in (
        MouthStateAnalyzer.landmark_indices()
    ):

        landmark = face[index]

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            3,
            (255, 120, 255),
            -1,
        )


# ============================================================
# Dashboard utilities
# ============================================================

def state_color(
    state: DriverState,
):
    if state == DriverState.NORMAL:
        return (80, 220, 80)

    if state == DriverState.WARMING_UP:
        return (0, 220, 255)

    if state == DriverState.SUSPECTED:
        return (0, 165, 255)

    if state == DriverState.DROWSY:
        return (0, 0, 255)

    return (180, 180, 180)


def presence_color(
    state: DriverPresence,
):
    if state == DriverPresence.PRESENT:
        return (80, 220, 80)

    if state == DriverPresence.ABSENT:
        return (0, 0, 255)

    return (0, 220, 255)


def draw_text(
    frame,
    text: str,
    x: int,
    y: int,
    scale: float = 0.60,
    color=(230, 230, 230),
    thickness: int = 1,
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


def draw_cabin_dashboard(
    frame,
    presence_result,
    driver_state_result,
    face_visible: bool,
    eye_closed: Optional[bool],
    current_yawn: bool,
    blink_count: int,
    recommendation,
):
    """
    Compact high-level Cabin Perception dashboard.

    The main product-facing outputs are:

        Driver Presence
        Driver State
        Risk

    PERCLOS / closure / yawns are retained only as
    interpretable fatigue evidence.
    """

    height, width = (
        frame.shape[:2]
    )

    panel_width = 390

    x1 = max(
        0,
        width - panel_width,
    )

    x2 = width

    # --------------------------------------------------------
    # Semi-transparent panel
    # --------------------------------------------------------

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,
        (x1, 0),
        (x2, height),
        (18, 18, 18),
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
        x1 + 25
    )

    y = 45

    # ========================================================
    # Header
    # ========================================================

    draw_text(
        frame,
        "VEHICLEMIND",
        left,
        y,
        scale=0.82,
        color=(255, 255, 255),
        thickness=2,
    )

    y += 30

    draw_text(
        frame,
        "CABIN PERCEPTION",
        left,
        y,
        scale=0.52,
        color=(170, 170, 170),
    )

    y += 35

    cv2.line(
        frame,
        (left, y),
        (width - 25, y),
        (90, 90, 90),
        1,
    )

    y += 35

    # ========================================================
    # Driver
    # ========================================================

    draw_text(
        frame,
        "DRIVER",
        left,
        y,
        scale=0.55,
        color=(160, 160, 160),
        thickness=2,
    )

    y += 35

    p_state = (
        presence_result.state
    )

    draw_text(
        frame,
        "Presence",
        left,
        y,
        scale=0.58,
    )

    draw_text(
        frame,
        p_state.value,
        left + 180,
        y,
        scale=0.65,
        color=presence_color(
            p_state
        ),
        thickness=2,
    )

    y += 35

    # --------------------------------------------------------
    # Observation status
    # --------------------------------------------------------

    if face_visible:

        observation_text = (
            "TRACKING"
        )

        observation_color = (
            80,
            220,
            80,
        )

    elif (
        p_state
        == DriverPresence.PRESENT
    ):

        observation_text = (
            "TEMP LOST"
        )

        observation_color = (
            0,
            200,
            255,
        )

    else:

        observation_text = (
            "NO FACE"
        )

        observation_color = (
            170,
            170,
            170,
        )

    draw_text(
        frame,
        "Observation",
        left,
        y,
        scale=0.58,
    )

    draw_text(
        frame,
        observation_text,
        left + 180,
        y,
        scale=0.58,
        color=observation_color,
        thickness=2,
    )

    y += 45

    # ========================================================
    # Driver State
    # ========================================================

    draw_text(
        frame,
        "DRIVER STATE",
        left,
        y,
        scale=0.55,
        color=(160, 160, 160),
        thickness=2,
    )

    y += 35

    if driver_state_result is None:

        state = (
            DriverState.UNKNOWN
        )

        risk_text = (
            "UNKNOWN"
        )

    else:

        state = (
            driver_state_result.state
        )

        risk_text = (
            driver_state_result
            .risk_level
            .value
        )

    draw_text(
        frame,
        "State",
        left,
        y,
        scale=0.58,
    )

    draw_text(
        frame,
        state.value,
        left + 180,
        y,
        scale=0.65,
        color=state_color(
            state
        ),
        thickness=2,
    )

    y += 35

    draw_text(
        frame,
        "Risk",
        left,
        y,
        scale=0.58,
    )

    if risk_text == "LOW":

        risk_color = (
            80,
            220,
            80,
        )

    elif risk_text == "MEDIUM":

        risk_color = (
            0,
            165,
            255,
        )

    elif risk_text == "HIGH":

        risk_color = (
            0,
            0,
            255,
        )

    else:

        risk_color = (
            170,
            170,
            170,
        )

    draw_text(
        frame,
        risk_text,
        left + 180,
        y,
        scale=0.65,
        color=risk_color,
        thickness=2,
    )

    y += 45

    # ========================================================
    # Fatigue Evidence
    # ========================================================

    draw_text(
        frame,
        "FATIGUE EVIDENCE",
        left,
        y,
        scale=0.55,
        color=(160, 160, 160),
        thickness=2,
    )

    y += 35

    # --------------------------------------------------------
    # Driver unavailable
    # --------------------------------------------------------

    if (
        driver_state_result is None
        or
        p_state != DriverPresence.PRESENT
    ):

        draw_text(
            frame,
            "PERCLOS",
            left,
            y,
        )

        draw_text(
            frame,
            "--",
            left + 180,
            y,
        )

        y += 32

        draw_text(
            frame,
            "Eye Closure",
            left,
            y,
        )

        draw_text(
            frame,
            "--",
            left + 180,
            y,
        )

        y += 32

        draw_text(
            frame,
            "Recent Yawns",
            left,
            y,
        )

        draw_text(
            frame,
            "--",
            left + 180,
            y,
        )

    else:

        # ----------------------------------------------------
        # PERCLOS
        # ----------------------------------------------------

        draw_text(
            frame,
            "PERCLOS",
            left,
            y,
        )

        if (
            driver_state_result
            .perclos_ready
        ):

            perclos_text = (
                f"{driver_state_result.perclos * 100:.1f}%"
            )

        else:

            perclos_text = (
                "WARMING UP"
            )

        draw_text(
            frame,
            perclos_text,
            left + 180,
            y,
            color=(220, 220, 220),
        )

        y += 32

        # ----------------------------------------------------
        # Continuous closure
        # ----------------------------------------------------

        draw_text(
            frame,
            "Eye Closure",
            left,
            y,
        )

        draw_text(
            frame,
            (
                f"{driver_state_result.continuous_eye_closure:.1f}s"
            ),
            left + 180,
            y,
        )

        y += 32

        # ----------------------------------------------------
        # Yawn
        # ----------------------------------------------------

        draw_text(
            frame,
            "Recent Yawns",
            left,
            y,
        )

        draw_text(
            frame,
            str(
                driver_state_result
                .recent_yawns
            ),
            left + 180,
            y,
        )

    y += 45

    # ========================================================
    # Monitoring
    # ========================================================

    draw_text(
        frame,
        "MONITORING",
        left,
        y,
        scale=0.55,
        color=(160, 160, 160),
        thickness=2,
    )

    y += 32

    # --------------------------------------------------------
    # Eye state
    # --------------------------------------------------------

    if eye_closed is None:

        eye_text = (
            "--"
        )

        eye_color = (
            170,
            170,
            170,
        )

    elif eye_closed:

        eye_text = (
            "CLOSED"
        )

        eye_color = (
            0,
            0,
            255,
        )

    else:

        eye_text = (
            "OPEN"
        )

        eye_color = (
            80,
            220,
            80,
        )

    draw_text(
        frame,
        "Eye",
        left,
        y,
    )

    draw_text(
        frame,
        eye_text,
        left + 180,
        y,
        color=eye_color,
        thickness=2,
    )

    y += 30

    # --------------------------------------------------------
    # Blink
    # --------------------------------------------------------

    draw_text(
        frame,
        "Blinks",
        left,
        y,
    )

    draw_text(
        frame,
        str(
            blink_count
        ),
        left + 180,
        y,
    )

    y += 30

    # --------------------------------------------------------
    # Current yawn
    # --------------------------------------------------------

    draw_text(
        frame,
        "Yawn",
        left,
        y,
    )

    if current_yawn:

        draw_text(
            frame,
            "DETECTED",
            left + 180,
            y,
            color=(0, 165, 255),
            thickness=2,
        )

    else:

        draw_text(
            frame,
            "NO",
            left + 180,
            y,
        )

    y += 40

    # ========================================================
    # Safety Assistant
    # ========================================================

    cv2.line(
        frame,
        (left, y),
        (width - 25, y),
        (90, 90, 90),
        1,
    )

    y += 30

    draw_text(
        frame,
        "SAFETY ASSISTANT",
        left,
        y,
        scale=0.55,
        color=(160, 160, 160),
        thickness=2,
    )

    y += 32

    if state == DriverState.DROWSY:

        draw_text(
            frame,
            "FATIGUE WARNING",
            left,
            y,
            scale=0.68,
            color=(0, 0, 255),
            thickness=2,
        )

        y += 30

        draw_text(
            frame,
            "Consider taking a break.",
            left,
            y,
            scale=0.52,
            color=(230, 230, 230),
        )

        y += 28

        if recommendation is not None:

            draw_text(
                frame,
                recommendation.name,
                left,
                y,
                scale=0.52,
                color=(255, 255, 255),
                thickness=2,
            )

            y += 25

            draw_text(
                frame,
                (
                    f"{recommendation.distance_km:.1f} km"
                    f"  |  ETA {recommendation.eta_minutes} min"
                ),
                left,
                y,
                scale=0.50,
                color=(200, 200, 200),
            )

    elif state == DriverState.SUSPECTED:

        draw_text(
            frame,
            "Fatigue signs detected.",
            left,
            y,
            scale=0.55,
            color=(0, 165, 255),
            thickness=2,
        )

    elif state == DriverState.NORMAL:

        draw_text(
            frame,
            "Driver condition normal.",
            left,
            y,
            scale=0.55,
            color=(80, 220, 80),
        )

    elif state == DriverState.WARMING_UP:

        draw_text(
            frame,
            "Collecting observations...",
            left,
            y,
            scale=0.52,
            color=(0, 220, 255),
        )

    else:

        draw_text(
            frame,
            "Driver unavailable.",
            left,
            y,
            scale=0.52,
            color=(170, 170, 170),
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    # ========================================================
    # 1. Face perception
    # ========================================================

    face_detector = (
        FaceLandmarkDetector(
            model_path=MODEL_PATH,
        )
    )

    # ========================================================
    # 2. Driver Presence
    # ========================================================

    presence_tracker = (
        DriverPresenceTracker(
            present_confirm_seconds=0.15,
            absence_timeout_seconds=1.5,
            startup_timeout_seconds=1.0,
        )
    )

    # ========================================================
    # 3. Eye monitoring
    # ========================================================

    eye_analyzer = (
        EyeStateAnalyzer(
            ear_threshold=0.21,
        )
    )

    blink_detector = (
        BlinkDetector(
            min_closed_frames=2,
            max_closed_frames=15,
        )
    )

    perclos_estimator = (
        PerclosEstimator(
            window_seconds=30.0,
            min_observation_seconds=5.0,
        )
    )

    # ========================================================
    # 4. Mouth / Yawn
    # ========================================================

    mouth_analyzer = (
        MouthStateAnalyzer(
            mar_threshold=0.35,
        )
    )

    yawn_detector = (
        YawnDetector(
            min_open_seconds=1.2,
        )
    )

    # ========================================================
    # 5. Driver State
    # ========================================================

    driver_state_estimator = (
        DriverStateEstimator(
            suspected_perclos=0.25,
            drowsy_perclos=0.30,
            suspected_closure_seconds=1.2,
            drowsy_closure_seconds=2.0,
            yawn_window_seconds=60.0,
            suspected_yawns=2,
        )
    )

    # ========================================================
    # 6. Safety Assistant
    # ========================================================

    rest_advisor = (
        MockRestAdvisor()
    )

    # ========================================================
    # 7. Camera
    # ========================================================

    cap = open_camera(
        0
    )

    start_time = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = 0.0
    fps_counter = 0

    fps_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Persistent state
    # --------------------------------------------------------

    last_driver_state_result = None

    last_yawn_count = 0

    last_blink_count = 0

    print(
        "[VehicleMind] Cabin Perception started."
    )

    print(
        "[VehicleMind] "
        "Outputs: Driver Presence + Driver State"
    )

    print(
        "Press 'q' to quit."
    )

    try:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:

                print(
                    "[VehicleMind] "
                    "Failed to read camera frame."
                )

                break

            # =================================================
            # Mirror front camera
            # =================================================

            frame = cv2.flip(
                frame,
                1,
            )

            # =================================================
            # Timestamp
            # =================================================

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            # =================================================
            # FPS
            # =================================================

            fps_counter += 1

            fps_elapsed = (
                time.perf_counter()
                - fps_start
            )

            if fps_elapsed >= 1.0:

                fps = (
                    fps_counter
                    / fps_elapsed
                )

                fps_counter = 0

                fps_start = (
                    time.perf_counter()
                )

            # =================================================
            # Face perception
            # =================================================

            faces = (
                face_detector.detect(
                    frame,
                    timestamp_ms,
                )
            )

            face_visible = (
                len(faces) > 0
            )

            # =================================================
            # Driver Presence
            # =================================================

            presence_result = (
                presence_tracker.update(
                    timestamp_ms=timestamp_ms,
                    face_detected=face_visible,
                )
            )

            # =================================================
            # Current-frame evidence
            # =================================================

            eye_closed_now = None

            yawn_now = False

            recommendation = None

            # =================================================
            # Valid face observation
            # =================================================

            if face_visible:

                face = (
                    faces[0]
                )

                height, width = (
                    frame.shape[:2]
                )

                # -------------------------------------------------
                # Visualization
                # -------------------------------------------------

                draw_face_landmarks(
                    frame,
                    face,
                )

                draw_eye_landmarks(
                    frame,
                    face,
                )

                draw_mouth_landmarks(
                    frame,
                    face,
                )

                # =================================================
                # Eye
                # =================================================

                eye_result = (
                    eye_analyzer.analyze(
                        face,
                        width,
                        height,
                    )
                )

                eye_closed_now = (
                    eye_result.is_closed
                )

                # =================================================
                # Blink
                # =================================================

                blink_result = (
                    blink_detector.update(
                        eye_result.is_closed
                    )
                )

                last_blink_count = (
                    blink_result.blink_count
                )

                # =================================================
                # PERCLOS
                # =================================================

                perclos_result = (
                    perclos_estimator.update(
                        timestamp_ms,
                        eye_result.is_closed,
                    )
                )

                # =================================================
                # Mouth
                # =================================================

                mouth_result = (
                    mouth_analyzer.analyze(
                        face,
                        width,
                        height,
                    )
                )

                # =================================================
                # Yawn
                # =================================================

                yawn_result = (
                    yawn_detector.update(
                        timestamp_ms,
                        mouth_result.is_open,
                    )
                )

                yawn_now = (
                    yawn_result.is_yawning
                )

                last_yawn_count = (
                    yawn_result.yawn_count
                )

                # =================================================
                # Driver State
                # =================================================

                driver_state_result = (
                    driver_state_estimator.update(
                        timestamp_ms=timestamp_ms,
                        driver_presence=(
                            presence_result.state
                        ),
                        eye_closed=(
                            eye_result.is_closed
                        ),
                        perclos=(
                            perclos_result.perclos
                        ),
                        perclos_ready=(
                            perclos_result.ready
                        ),
                        yawn_count=(
                            yawn_result.yawn_count
                        ),
                    )
                )

                last_driver_state_result = (
                    driver_state_result
                )

            # =================================================
            # Face unavailable
            # =================================================

            else:

                # -------------------------------------------------
                # Tell PERCLOS that this interval is invalid.
                #
                # No face does NOT mean eyes are open.
                # -------------------------------------------------

                perclos_result = (
                    perclos_estimator.update(
                        timestamp_ms,
                        None,
                    )
                )

                # -------------------------------------------------
                # Short detector dropout:
                #
                # Presence Tracker still considers the driver
                # PRESENT. Keep the last reliable Driver State
                # instead of feeding fake "eyes open" evidence.
                # -------------------------------------------------

                if (
                    presence_result.state
                    == DriverPresence.PRESENT
                ):

                    driver_state_result = (
                        last_driver_state_result
                    )

                # -------------------------------------------------
                # Driver truly unavailable:
                #
                # Force Driver State -> UNKNOWN.
                # -------------------------------------------------

                else:

                    driver_state_result = (
                        driver_state_estimator.update(
                            timestamp_ms=timestamp_ms,
                            driver_presence=(
                                presence_result.state
                            ),
                            eye_closed=False,
                            perclos=(
                                perclos_result.perclos
                            ),
                            perclos_ready=(
                                perclos_result.ready
                            ),
                            yawn_count=(
                                last_yawn_count
                            ),
                        )
                    )

                    last_driver_state_result = (
                        driver_state_result
                    )

            # =================================================
            # Safety Assistant
            # =================================================

            if (
                driver_state_result is not None
                and
                driver_state_result.state
                == DriverState.DROWSY
                and
                presence_result.state
                == DriverPresence.PRESENT
            ):

                recommendation = (
                    rest_advisor.recommend()
                )

            # =================================================
            # Minimal system information
            # =================================================

            draw_text(
                frame,
                (
                    f"FPS {fps:.1f}"
                ),
                20,
                35,
                scale=0.62,
                color=(220, 220, 220),
                thickness=2,
            )

            # =================================================
            # Cabin Dashboard
            # =================================================

            draw_cabin_dashboard(
                frame=frame,
                presence_result=presence_result,
                driver_state_result=(
                    driver_state_result
                ),
                face_visible=face_visible,
                eye_closed=eye_closed_now,
                current_yawn=yawn_now,
                blink_count=last_blink_count,
                recommendation=recommendation,
            )

            # =================================================
            # Display
            # =================================================

            cv2.imshow(
                "VehicleMind - Cabin Perception",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        cap.release()

        face_detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
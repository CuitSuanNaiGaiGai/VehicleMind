import time
from pathlib import Path

import cv2

from modules.cabin.face.landmarks import FaceLandmarkDetector

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

from modules.cabin.head_pose.estimator import (
    HeadPoseEstimator,
)

from modules.cabin.state.driver_state import (
    DriverState,
    DriverStateEstimator,
)

from modules.cabin.assistance.rest_advisor import (
    MockRestAdvisor,
)

from modules.cabin.common.visualizer import (
    draw_driver_status_panel,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

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
    Open MacBook camera.

    AVFoundation is preferred on macOS.
    """

    cap = cv2.VideoCapture(
        camera_id,
        cv2.CAP_AVFOUNDATION,
    )

    # Fallback.
    if not cap.isOpened():
        cap = cv2.VideoCapture(
            camera_id
        )

    if not cap.isOpened():

        raise RuntimeError(
            "Failed to open camera. "
            "Please check macOS camera permission."
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
# Visualization utilities
# ============================================================

def draw_face_landmarks(
    frame,
    face_landmarks,
) -> None:
    """
    Draw all MediaPipe facial landmarks.
    """

    height, width = (
        frame.shape[:2]
    )

    for face in face_landmarks:

        for landmark in face:

            x = int(
                landmark.x
                * width
            )

            y = int(
                landmark.y
                * height
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
                    (0, 255, 0),
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
            landmark.x
            * width
        )

        y = int(
            landmark.y
            * height
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
    Highlight landmarks used for MAR.
    """

    height, width = (
        frame.shape[:2]
    )

    for index in (
        MouthStateAnalyzer.landmark_indices()
    ):

        landmark = face[index]

        x = int(
            landmark.x
            * width
        )

        y = int(
            landmark.y
            * height
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 0, 255),
            -1,
        )


def draw_head_pose_landmarks(
    frame,
    face,
) -> None:
    """
    Draw landmarks used by solvePnP head-pose estimation.
    """

    height, width = (
        frame.shape[:2]
    )

    for index in (
        HeadPoseEstimator.landmark_indices()
    ):

        landmark = face[index]

        x = int(
            landmark.x
            * width
        )

        y = int(
            landmark.y
            * height
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 128, 0),
            -1,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    # ========================================================
    # 1. Face landmark detector
    # ========================================================

    detector = (
        FaceLandmarkDetector(
            model_path=MODEL_PATH,
        )
    )

    # ========================================================
    # 2. Eye analysis
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
    # 3. Mouth / Yawn analysis
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
    # 4. Head Pose
    # ========================================================

    head_pose_estimator = (
        HeadPoseEstimator()
    )

    # ========================================================
    # 5. Driver-state estimation
    # ========================================================

    driver_state_estimator = (
        DriverStateEstimator(
            suspected_perclos=0.15,
            drowsy_perclos=0.30,
            suspected_closure_seconds=1.2,
            drowsy_closure_seconds=2.5,
        )
    )

    # ========================================================
    # 6. Safety assistance
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

    fps = 0.0
    frame_count = 0

    fps_start = (
        time.perf_counter()
    )

    print(
        "Camera started."
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
                    "Failed to read frame."
                )

                break

            # ------------------------------------------------
            # Mirror MacBook front-facing camera
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1,
            )

            # ------------------------------------------------
            # Runtime timestamp
            # ------------------------------------------------

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            # =================================================
            # Face detection / landmark estimation
            # =================================================

            faces = detector.detect(
                frame,
                timestamp_ms,
            )

            draw_face_landmarks(
                frame,
                faces,
            )

            # =================================================
            # FPS
            # =================================================

            frame_count += 1

            elapsed = (
                time.perf_counter()
                - fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    frame_count
                    / elapsed
                )

                frame_count = 0

                fps_start = (
                    time.perf_counter()
                )

            # =================================================
            # Reset frame-level results
            # =================================================

            driver_state_result = None
            recommendation = None

            # =================================================
            # Driver perception
            # =================================================

            if len(faces) > 0:

                face = faces[0]

                height, width = (
                    frame.shape[:2]
                )

                # =============================================
                # Head Pose
                # =============================================

                head_pose_result = (
                    head_pose_estimator.estimate(
                        face,
                        width,
                        height,
                    )
                )

                # =============================================
                # Eye State
                # =============================================

                eye_result = (
                    eye_analyzer.analyze(
                        face,
                        width,
                        height,
                    )
                )

                # =============================================
                # Blink
                # =============================================

                blink_result = (
                    blink_detector.update(
                        eye_result.is_closed
                    )
                )

                # =============================================
                # PERCLOS
                # =============================================

                perclos_result = (
                    perclos_estimator.update(
                        timestamp_ms,
                        eye_result.is_closed,
                    )
                )

                # =============================================
                # Mouth State
                # =============================================

                mouth_result = (
                    mouth_analyzer.analyze(
                        face,
                        width,
                        height,
                    )
                )

                # =============================================
                # Yawn
                # =============================================

                yawn_result = (
                    yawn_detector.update(
                        timestamp_ms,
                        mouth_result.is_open,
                    )
                )

                # =============================================
                # Driver State
                # =============================================

                driver_state_result = (
                    driver_state_estimator.update(
                        timestamp_ms=timestamp_ms,
                        eye_closed=(
                            eye_result.is_closed
                        ),
                        perclos=(
                            perclos_result.perclos
                        ),
                        perclos_ready=(
                            perclos_result.ready
                        ),
                    )
                )

                # =============================================
                # Safety Assistance
                # =============================================

                if (
                    driver_state_result.state
                    == DriverState.DROWSY
                ):

                    recommendation = (
                        rest_advisor.recommend()
                    )

                # =============================================
                # Landmark Visualization
                # =============================================

                draw_eye_landmarks(
                    frame,
                    face,
                )

                draw_mouth_landmarks(
                    frame,
                    face,
                )

                draw_head_pose_landmarks(
                    frame,
                    face,
                )

                # =============================================
                # Eye state text
                # =============================================

                eye_state = (
                    "CLOSED"
                    if eye_result.is_closed
                    else "OPEN"
                )

                cv2.putText(
                    frame,
                    (
                        f"Left EAR: "
                        f"{eye_result.left_ear:.3f}"
                    ),
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Right EAR: "
                        f"{eye_result.right_ear:.3f}"
                    ),
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Mean EAR: "
                        f"{eye_result.mean_ear:.3f}"
                    ),
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Eye State: "
                        f"{eye_state}"
                    ),
                    (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (
                        (0, 0, 255)
                        if eye_result.is_closed
                        else (0, 255, 0)
                    ),
                    2,
                )

                # =============================================
                # Blink
                # =============================================

                cv2.putText(
                    frame,
                    (
                        f"Blinks: "
                        f"{blink_result.blink_count}"
                    ),
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        "Closed Frames: "
                        f"{blink_result.closed_frames}"
                    ),
                    (20, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                # =============================================
                # PERCLOS
                # =============================================

                if (
                    perclos_result.ready
                ):

                    perclos_text = (
                        "PERCLOS: "
                        f"{perclos_result.perclos * 100:.1f}%"
                    )

                else:

                    perclos_text = (
                        "PERCLOS: warming up "
                        f"("
                        f"{perclos_result.observed_duration:.1f}s"
                        f")"
                    )

                cv2.putText(
                    frame,
                    perclos_text,
                    (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

                # =============================================
                # Mouth
                # =============================================

                mouth_state = (
                    "OPEN"
                    if mouth_result.is_open
                    else "CLOSED"
                )

                cv2.putText(
                    frame,
                    (
                        f"MAR: "
                        f"{mouth_result.mar:.3f}"
                    ),
                    (20, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Mouth: "
                        f"{mouth_state}"
                    ),
                    (20, 385),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (
                        (0, 165, 255)
                        if mouth_result.is_open
                        else (0, 255, 0)
                    ),
                    2,
                )

                # =============================================
                # Yawn
                # =============================================

                cv2.putText(
                    frame,
                    (
                        f"Yawn Count: "
                        f"{yawn_result.yawn_count}"
                    ),
                    (20, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        "Mouth Open: "
                        f"{yawn_result.open_duration:.1f}s"
                    ),
                    (20, 455),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                if (
                    yawn_result.is_yawning
                ):

                    cv2.putText(
                        frame,
                        "YAWN DETECTED",
                        (20, 500),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                # =============================================
                # Head Pose
                # =============================================

                if (
                    head_pose_result.success
                ):

                    cv2.putText(
                        frame,
                        (
                            "Yaw: "
                            f"{head_pose_result.yaw:.1f} deg"
                        ),
                        (20, 545),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            "Pitch: "
                            f"{head_pose_result.pitch:.1f} deg"
                        ),
                        (20, 580),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        (
                            "Roll: "
                            f"{head_pose_result.roll:.1f} deg"
                        ),
                        (20, 615),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                else:

                    cv2.putText(
                        frame,
                        "Head Pose: unavailable",
                        (20, 545),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2,
                    )

            else:

                # =================================================
                # No reliable face observation
                # =================================================

                perclos_estimator.update(
                    timestamp_ms,
                    None,
                )

            # =================================================
            # Basic system information
            # =================================================

            cv2.putText(
                frame,
                (
                    f"Faces: "
                    f"{len(faces)}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                (
                    f"FPS: "
                    f"{fps:.1f}"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # =================================================
            # High-level Dashboard
            # =================================================

            if (
                driver_state_result
                is not None
            ):

                draw_driver_status_panel(
                    frame,
                    driver_state_result,
                    recommendation,
                )

            # =================================================
            # Display
            # =================================================

            cv2.imshow(
                "VehicleMind - Cabin Intelligence",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if (
                key == ord("q")
            ):

                break

    finally:

        cap.release()

        detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
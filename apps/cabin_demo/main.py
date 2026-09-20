import time
from pathlib import Path

import cv2

from modules.cabin.face.landmarks import FaceLandmarkDetector
from modules.cabin.fatigue.eye_state import EyeStateAnalyzer
from modules.cabin.fatigue.blink import BlinkDetector
from modules.cabin.fatigue.perclos import PerclosEstimator
from modules.cabin.fatigue.mouth_state import MouthStateAnalyzer
from modules.cabin.fatigue.yawn import YawnDetector

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Open MacBook camera."""

    cap = cv2.VideoCapture(
        camera_id,
        cv2.CAP_AVFOUNDATION,
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

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


def draw_face_landmarks(
    frame,
    face_landmarks,
) -> None:
    """Draw all MediaPipe facial landmarks."""

    height, width = frame.shape[:2]

    for face in face_landmarks:
        for landmark in face:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            if 0 <= x < width and 0 <= y < height:
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
    """Highlight landmarks used for EAR."""

    height, width = frame.shape[:2]

    for index in EyeStateAnalyzer.eye_indices():
        landmark = face[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

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
    """Highlight landmarks used for MAR."""

    height, width = frame.shape[:2]

    for index in MouthStateAnalyzer.landmark_indices():

        landmark = face[index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 0, 255),
            -1,
        )


def main() -> None:

    # ==========================================================
    # 1. Initialize perception modules
    # ==========================================================

    detector = FaceLandmarkDetector(
        model_path=MODEL_PATH,
    )

    eye_analyzer = EyeStateAnalyzer(
        ear_threshold=0.21,
    )

    blink_detector = BlinkDetector(
        min_closed_frames=2,
        max_closed_frames=15,
    )

    perclos_estimator = PerclosEstimator(
        window_seconds=30.0,
        min_observation_seconds=5.0,
    )

    mouth_analyzer = MouthStateAnalyzer(
        mar_threshold=0.35,
    )

    yawn_detector = YawnDetector(
        min_open_seconds=1.2,
    )

    # ==========================================================
    # 2. Initialize driver-state reasoning
    # ==========================================================

    driver_state_estimator = DriverStateEstimator(
        suspected_perclos=0.15,
        drowsy_perclos=0.30,
        suspected_closure_seconds=1.2,
        drowsy_closure_seconds=2.5,
    )

    # ==========================================================
    # 3. Safety assistance
    # ==========================================================

    rest_advisor = MockRestAdvisor()

    # ==========================================================
    # 4. Camera
    # ==========================================================

    cap = open_camera(0)

    start_time = time.perf_counter()

    fps = 0.0
    frame_count = 0
    fps_start = time.perf_counter()

    print("Camera started.")
    print("Press 'q' to quit.")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Failed to read frame.")
                break

            # Mirror MacBook front-facing camera.
            frame = cv2.flip(frame, 1)

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            # ==================================================
            # Face detection / landmark estimation
            # ==================================================

            faces = detector.detect(
                frame,
                timestamp_ms,
            )

            draw_face_landmarks(
                frame,
                faces,
            )

            # ==================================================
            # FPS
            # ==================================================

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
                fps_start = time.perf_counter()

            # ==================================================
            # Reset per-frame high-level outputs
            # ==================================================

            driver_state_result = None
            recommendation = None

            # ==================================================
            # Driver perception
            # ==================================================

            if len(faces) > 0:

                face = faces[0]

                height, width = frame.shape[:2]

                # ----------------------------------------------
                # Eye state
                # ----------------------------------------------

                eye_result = eye_analyzer.analyze(
                    face,
                    width,
                    height,
                )

                blink_result = blink_detector.update(
                    eye_result.is_closed,
                )

                perclos_result = perclos_estimator.update(
                    timestamp_ms,
                    eye_result.is_closed,
                )

                # ----------------------------------------------
                # Mouth state
                # ----------------------------------------------

                mouth_result = mouth_analyzer.analyze(
                    face,
                    width,
                    height,
                )

                yawn_result = yawn_detector.update(
                    timestamp_ms,
                    mouth_result.is_open,
                )

                # ----------------------------------------------
                # Driver-state estimation
                #
                # Important:
                # Calculate the state BEFORE accessing it.
                # ----------------------------------------------

                driver_state_result = (
                    driver_state_estimator.update(
                        timestamp_ms=timestamp_ms,
                        eye_closed=eye_result.is_closed,
                        perclos=perclos_result.perclos,
                        perclos_ready=perclos_result.ready,
                    )
                )

                # ----------------------------------------------
                # Safety assistance
                # ----------------------------------------------

                if (
                    driver_state_result.state
                    == DriverState.DROWSY
                ):
                    recommendation = (
                        rest_advisor.recommend()
                    )

                # ==================================================
                # Debug visualization
                # ==================================================

                draw_eye_landmarks(
                    frame,
                    face,
                )

                draw_mouth_landmarks(
                    frame,
                    face,
                )

                eye_state = (
                    "CLOSED"
                    if eye_result.is_closed
                    else "OPEN"
                )

                mouth_state = (
                    "OPEN"
                    if mouth_result.is_open
                    else "CLOSED"
                )

                # ----------------------------------------------
                # Eye metrics
                # ----------------------------------------------

                cv2.putText(
                    frame,
                    f"Left EAR: {eye_result.left_ear:.3f}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Right EAR: {eye_result.right_ear:.3f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Mean EAR: {eye_result.mean_ear:.3f}",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Eye State: {eye_state}",
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

                # ----------------------------------------------
                # Blink metrics
                # ----------------------------------------------

                cv2.putText(
                    frame,
                    f"Blinks: {blink_result.blink_count}",
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

                # ----------------------------------------------
                # PERCLOS
                # ----------------------------------------------

                if perclos_result.ready:

                    perclos_text = (
                        "PERCLOS: "
                        f"{perclos_result.perclos * 100:.1f}%"
                    )

                else:

                    perclos_text = (
                        "PERCLOS: warming up "
                        f"({perclos_result.observed_duration:.1f}s)"
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

                # ----------------------------------------------
                # Mouth / Yawn
                # ----------------------------------------------

                cv2.putText(
                    frame,
                    f"MAR: {mouth_result.mar:.3f}",
                    (20, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Mouth: {mouth_state}",
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

                cv2.putText(
                    frame,
                    (
                        "Yawn Count: "
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

                if yawn_result.is_yawning:

                    cv2.putText(
                        frame,
                        "YAWN DETECTED",
                        (20, 500),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

            else:

                # No reliable face/eye observation.
                # This duration must not be counted as
                # open-eye or closed-eye time.

                perclos_estimator.update(
                    timestamp_ms,
                    None,
                )

            # ==================================================
            # Basic system information
            # ==================================================

            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # ==================================================
            # High-level VehicleMind dashboard
            # ==================================================

            if driver_state_result is not None:

                draw_driver_status_panel(
                    frame,
                    driver_state_result,
                    recommendation,
                )

            # ==================================================
            # Display
            # ==================================================

            cv2.imshow(
                "VehicleMind - Cabin Intelligence",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
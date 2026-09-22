import time
from pathlib import Path

import cv2

from apps.cabin_demo.ui.cabin import (
    draw_cabin_dashboard,
    draw_eye_landmarks,
    draw_face_landmarks,
    draw_mouth_landmarks,
    draw_text,
)
from modules.config import CabinPerceptionConfig
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


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
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera. Please check camera permission.")

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    return cap


def main() -> None:
    config = CabinPerceptionConfig.load_default()

    # ========================================================
    # 1. Face perception
    # ========================================================

    face_detector = FaceLandmarkDetector(
        model_path=MODEL_PATH,
    )

    # ========================================================
    # 2. Driver Presence
    # ========================================================

    presence_tracker = DriverPresenceTracker(
        present_confirm_seconds=config.presence.present_confirm_seconds,
        absence_timeout_seconds=config.presence.absence_timeout_seconds,
        startup_timeout_seconds=config.presence.startup_timeout_seconds,
    )

    # ========================================================
    # 3. Eye monitoring
    # ========================================================

    eye_analyzer = EyeStateAnalyzer(
        ear_threshold=config.eye.ear_threshold,
    )

    blink_detector = BlinkDetector(
        min_closed_frames=config.blink.min_closed_frames,
        max_closed_frames=config.blink.max_closed_frames,
    )

    perclos_estimator = PerclosEstimator(
        window_seconds=config.perclos.window_seconds,
        min_observation_seconds=config.perclos.min_observation_seconds,
    )

    # ========================================================
    # 4. Mouth / Yawn
    # ========================================================

    mouth_analyzer = MouthStateAnalyzer(
        mar_threshold=config.mouth.mar_threshold,
    )

    yawn_detector = YawnDetector(
        min_open_seconds=config.yawn.min_open_seconds,
    )

    # ========================================================
    # 5. Driver State
    # ========================================================

    driver_state_estimator = DriverStateEstimator(
        suspected_perclos=config.driver_state.suspected_perclos,
        drowsy_perclos=config.driver_state.drowsy_perclos,
        suspected_closure_seconds=(config.driver_state.suspected_closure_seconds),
        drowsy_closure_seconds=config.driver_state.drowsy_closure_seconds,
        yawn_window_seconds=config.driver_state.yawn_window_seconds,
        suspected_yawns=config.driver_state.suspected_yawns,
    )

    # ========================================================
    # 6. Safety Assistant
    # ========================================================

    rest_advisor = MockRestAdvisor()

    # ========================================================
    # 7. Camera
    # ========================================================

    cap = open_camera(0)

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = 0.0
    fps_counter = 0

    fps_start = time.perf_counter()

    # --------------------------------------------------------
    # Persistent state
    # --------------------------------------------------------

    last_driver_state_result = None

    last_yawn_count = 0

    last_blink_count = 0

    print("[VehicleMind] Cabin Perception started.")

    print("[VehicleMind] Outputs: Driver Presence + Driver State")

    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("[VehicleMind] Failed to read camera frame.")

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

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)

            # =================================================
            # FPS
            # =================================================

            fps_counter += 1

            fps_elapsed = time.perf_counter() - fps_start

            if fps_elapsed >= 1.0:
                fps = fps_counter / fps_elapsed

                fps_counter = 0

                fps_start = time.perf_counter()

            # =================================================
            # Face perception
            # =================================================

            faces = face_detector.detect(
                frame,
                timestamp_ms,
            )

            face_visible = len(faces) > 0

            # =================================================
            # Driver Presence
            # =================================================

            presence_result = presence_tracker.update(
                timestamp_ms=timestamp_ms,
                face_detected=face_visible,
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
                face = faces[0]

                height, width = frame.shape[:2]

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

                eye_result = eye_analyzer.analyze(
                    face,
                    width,
                    height,
                )

                eye_closed_now = eye_result.is_closed

                # =================================================
                # Blink
                # =================================================

                blink_result = blink_detector.update(eye_result.is_closed)

                last_blink_count = blink_result.blink_count

                # =================================================
                # PERCLOS
                # =================================================

                perclos_result = perclos_estimator.update(
                    timestamp_ms,
                    eye_result.is_closed,
                )

                # =================================================
                # Mouth
                # =================================================

                mouth_result = mouth_analyzer.analyze(
                    face,
                    width,
                    height,
                )

                # =================================================
                # Yawn
                # =================================================

                yawn_result = yawn_detector.update(
                    timestamp_ms,
                    mouth_result.is_open,
                )

                yawn_now = yawn_result.is_yawning

                last_yawn_count = yawn_result.yawn_count

                # =================================================
                # Driver State
                # =================================================

                driver_state_result = driver_state_estimator.update(
                    timestamp_ms=timestamp_ms,
                    driver_presence=(presence_result.state),
                    eye_closed=(eye_result.is_closed),
                    perclos=(perclos_result.perclos),
                    perclos_ready=(perclos_result.ready),
                    yawn_count=(yawn_result.yawn_count),
                )

                last_driver_state_result = driver_state_result

            # =================================================
            # Face unavailable
            # =================================================

            else:
                # -------------------------------------------------
                # Tell PERCLOS that this interval is invalid.
                #
                # No face does NOT mean eyes are open.
                # -------------------------------------------------

                perclos_result = perclos_estimator.update(
                    timestamp_ms,
                    None,
                )

                # -------------------------------------------------
                # Short detector dropout:
                #
                # Presence Tracker still considers the driver
                # PRESENT. Keep the last reliable Driver State
                # instead of feeding fake "eyes open" evidence.
                # -------------------------------------------------

                if presence_result.state == DriverPresence.PRESENT:
                    driver_state_result = last_driver_state_result

                # -------------------------------------------------
                # Driver truly unavailable:
                #
                # Force Driver State -> UNKNOWN.
                # -------------------------------------------------

                else:
                    driver_state_result = driver_state_estimator.update(
                        timestamp_ms=timestamp_ms,
                        driver_presence=(presence_result.state),
                        eye_closed=False,
                        perclos=(perclos_result.perclos),
                        perclos_ready=(perclos_result.ready),
                        yawn_count=(last_yawn_count),
                    )

                    last_driver_state_result = driver_state_result

            # =================================================
            # Safety Assistant
            # =================================================

            if (
                driver_state_result is not None
                and driver_state_result.state == DriverState.DROWSY
                and presence_result.state == DriverPresence.PRESENT
            ):
                recommendation = rest_advisor.recommend()

            # =================================================
            # Minimal system information
            # =================================================

            draw_text(
                frame,
                (f"FPS {fps:.1f}"),
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
                driver_state_result=(driver_state_result),
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

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()

        face_detector.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

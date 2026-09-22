from pathlib import Path

import cv2

from apps.cabin_demo.ui.video import draw_demo_header, draw_eye_landmarks
from apps.cabin_demo.video_cli import parse_args
from apps.cabin_demo.video_diagnostics import print_cabin_diagnostics
from apps.cabin_demo.video_export import generate_gif_with_ffmpeg
from modules.config import CabinPerceptionConfig
from modules.cabin.face.landmarks import FaceLandmarkDetector
from modules.cabin.fatigue.eye_state import EyeStateAnalyzer
from modules.cabin.fatigue.blink import BlinkDetector
from modules.cabin.fatigue.perclos import PerclosEstimator

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


def main():

    args = parse_args()
    config = CabinPerceptionConfig.load_default()

    # ========================================================
    # Paths
    # ========================================================

    video_path = PROJECT_ROOT / args.video

    output_video_path = PROJECT_ROOT / args.output_video

    output_gif_path = PROJECT_ROOT / args.output_gif

    if not video_path.exists():
        raise FileNotFoundError(f"Demo video not found: {video_path}")

    output_video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_gif_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Open input video
    # ========================================================

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        source_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_duration = total_frames / source_fps if source_fps > 0 else 0.0

    print(f"[VehicleMind] Input: {video_path}")

    print(f"[VehicleMind] Resolution: {width}x{height}")

    print(f"[VehicleMind] FPS: {source_fps:.2f}")

    print(f"[VehicleMind] Frames: {total_frames}")

    print(f"[VehicleMind] Duration: {video_duration:.2f}s")

    # ========================================================
    # Output video writer
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        source_fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError("Failed to initialize output video writer.")

    # ========================================================
    # VehicleMind perception modules
    # ========================================================

    detector = FaceLandmarkDetector(
        model_path=MODEL_PATH,
    )

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

    # --------------------------------------------------------
    # Driver state thresholds
    #
    # These are demonstration / engineering thresholds.
    #
    # They are not claimed as production DMS,
    # medical, or traffic-safety thresholds.
    # --------------------------------------------------------

    driver_state_estimator = DriverStateEstimator(
        suspected_perclos=config.driver_state.suspected_perclos,
        drowsy_perclos=config.driver_state.drowsy_perclos,
        suspected_closure_seconds=(config.driver_state.suspected_closure_seconds),
        drowsy_closure_seconds=config.driver_state.drowsy_closure_seconds,
        yawn_window_seconds=config.driver_state.yawn_window_seconds,
        suspected_yawns=config.driver_state.suspected_yawns,
    )

    rest_advisor = MockRestAdvisor()

    # ========================================================
    # Diagnostics
    # ========================================================

    diag_ear_values = []

    diag_face_frames = 0

    diag_closed_frames = 0

    diag_max_perclos = 0.0

    diag_max_ready_perclos = 0.0

    diag_closed_start_ms = None

    diag_max_closure_ms = 0

    diag_suspected_frames = 0

    diag_drowsy_frames = 0

    # ========================================================
    # DROWSY event
    # ========================================================

    first_drowsy_event_time = None

    # ========================================================
    # Frame processing
    # ========================================================

    frame_index = 0

    previous_timestamp_ms = -1

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # ==================================================
            # Offline-video timestamp
            #
            # Temporal fatigue features MUST use video time,
            # rather than CPU / wall-clock processing time.
            # ==================================================

            timestamp_ms = int((frame_index / source_fps) * 1000)

            # MediaPipe VIDEO mode requires monotonically
            # increasing timestamps.
            timestamp_ms = max(
                timestamp_ms,
                previous_timestamp_ms + 1,
            )

            previous_timestamp_ms = timestamp_ms

            # ==================================================
            # Face landmarks
            # ==================================================

            faces = detector.detect(
                frame,
                timestamp_ms,
            )

            driver_state_result = None
            recommendation = None

            # ==================================================
            # Driver perception
            # ==================================================

            if faces:
                face = faces[0]

                diag_face_frames += 1

                # ----------------------------------------------
                # Eye state
                # ----------------------------------------------

                eye_result = eye_analyzer.analyze(
                    face,
                    width,
                    height,
                )

                diag_ear_values.append(eye_result.mean_ear)

                # ----------------------------------------------
                # Continuous closure diagnostics
                # ----------------------------------------------

                if eye_result.is_closed:
                    diag_closed_frames += 1

                    if diag_closed_start_ms is None:
                        diag_closed_start_ms = timestamp_ms

                    current_closure_ms = timestamp_ms - diag_closed_start_ms

                    diag_max_closure_ms = max(
                        diag_max_closure_ms,
                        current_closure_ms,
                    )

                else:
                    diag_closed_start_ms = None

                # ----------------------------------------------
                # Blink detection
                # ----------------------------------------------

                blink_detector.update(eye_result.is_closed)

                # ----------------------------------------------
                # PERCLOS
                # ----------------------------------------------

                perclos_result = perclos_estimator.update(
                    timestamp_ms,
                    eye_result.is_closed,
                )

                diag_max_perclos = max(
                    diag_max_perclos,
                    perclos_result.perclos,
                )

                if perclos_result.ready:
                    diag_max_ready_perclos = max(
                        diag_max_ready_perclos,
                        perclos_result.perclos,
                    )

                # ----------------------------------------------
                # Driver state
                # ----------------------------------------------

                driver_state_result = driver_state_estimator.update(
                    timestamp_ms=timestamp_ms,
                    eye_closed=(eye_result.is_closed),
                    perclos=(perclos_result.perclos),
                    perclos_ready=(perclos_result.ready),
                )

                # ----------------------------------------------
                # State diagnostics
                # ----------------------------------------------

                if driver_state_result.state == DriverState.SUSPECTED:
                    diag_suspected_frames += 1

                if driver_state_result.state == DriverState.DROWSY:
                    diag_drowsy_frames += 1

                    recommendation = rest_advisor.recommend()

                    # ------------------------------------------
                    # Record the FIRST DROWSY event only.
                    # ------------------------------------------

                    if first_drowsy_event_time is None:
                        first_drowsy_event_time = timestamp_ms / 1000.0

                        print(
                            "[VehicleMind] "
                            "DROWSY event detected at "
                            f"{first_drowsy_event_time:.2f}s"
                        )

                # ----------------------------------------------
                # Minimal eye visualization
                # ----------------------------------------------

                draw_eye_landmarks(
                    frame,
                    face,
                )

            else:
                # No reliable eye observation.
                #
                # Do not count missing-face duration as
                # open or closed eye time.
                perclos_estimator.update(
                    timestamp_ms,
                    None,
                )

                diag_closed_start_ms = None

            # ==================================================
            # VehicleMind visualization
            # ==================================================

            draw_demo_header(
                frame,
                source_fps,
            )

            if driver_state_result is not None:
                draw_driver_status_panel(
                    frame,
                    driver_state_result,
                    recommendation,
                )

            else:
                cv2.putText(
                    frame,
                    "Driver face unavailable",
                    (
                        25,
                        height - 30,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ==================================================
            # Write processed result
            # ==================================================

            writer.write(frame)

            # ==================================================
            # Optional preview
            # ==================================================

            if args.show:
                cv2.imshow(
                    "VehicleMind - Public Demo",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            frame_index += 1

    finally:
        cap.release()

        writer.release()

        detector.close()

        cv2.destroyAllWindows()

    print_cabin_diagnostics(
        output_video_path=output_video_path,
        eye_analyzer=eye_analyzer,
        diag_face_frames=diag_face_frames,
        frame_index=frame_index,
        diag_ear_values=diag_ear_values,
        diag_closed_frames=diag_closed_frames,
        diag_max_closure_ms=diag_max_closure_ms,
        diag_max_perclos=diag_max_perclos,
        diag_max_ready_perclos=diag_max_ready_perclos,
        diag_suspected_frames=diag_suspected_frames,
        diag_drowsy_frames=diag_drowsy_frames,
        first_drowsy_event_time=first_drowsy_event_time,
    )

    # ========================================================
    # Generate high-quality GIF with FFmpeg
    # ========================================================

    if first_drowsy_event_time is None:
        print()

        print("[VehicleMind] WARNING: No DROWSY event was detected.")

        print("[VehicleMind] GIF was not generated.")

        return

    generate_gif_with_ffmpeg(
        input_video=output_video_path,
        output_gif=output_gif_path,
        event_time=first_drowsy_event_time,
        pre_event=args.pre_event,
        post_event=args.post_event,
        gif_fps=max(
            1,
            args.gif_fps,
        ),
        gif_width=max(
            320,
            args.gif_width,
        ),
    )

    print()

    print("[VehicleMind] Demo generation complete.")


if __name__ == "__main__":
    main()

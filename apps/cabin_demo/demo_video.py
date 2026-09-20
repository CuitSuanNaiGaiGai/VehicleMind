import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="VehicleMind Cabin Intelligence Demo"
    )

    parser.add_argument(
        "--video",
        type=str,
        default="assets/demo/MicroSleep_test.mp4",
        help="Input public driver video",
    )

    parser.add_argument(
        "--output-video",
        type=str,
        default="assets/demo/MicroSleep_result.mp4",
        help="Processed result video",
    )

    parser.add_argument(
        "--output-gif",
        type=str,
        default="assets/demo/cabin_demo.gif",
        help="GitHub README GIF",
    )

    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="GIF frame rate",
    )

    parser.add_argument(
        "--gif-width",
        type=int,
        default=960,
        help="GIF width",
    )

    parser.add_argument(
        "--pre-event",
        type=float,
        default=3.0,
        help="Seconds retained before first DROWSY event",
    )

    parser.add_argument(
        "--post-event",
        type=float,
        default=5.0,
        help="Seconds retained after first DROWSY event",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization while processing",
    )

    return parser.parse_args()


# ============================================================
# Demo visualization
# ============================================================

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

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

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


# ============================================================
# FFmpeg GIF generation
# ============================================================

def check_ffmpeg() -> str:
    """
    Locate FFmpeg executable.
    """

    ffmpeg_path = shutil.which(
        "ffmpeg"
    )

    if ffmpeg_path is None:

        raise RuntimeError(
            "FFmpeg was not found.\n"
            "Install it first with:\n"
            "    brew install ffmpeg"
        )

    return ffmpeg_path


def generate_gif_with_ffmpeg(
    input_video: Path,
    output_gif: Path,
    event_time: float,
    pre_event: float,
    post_event: float,
    gif_fps: int,
    gif_width: int,
) -> None:
    """
    Generate a high-quality GIF from the processed MP4.

    FFmpeg creates one optimized global palette using
    palettegen and applies it using paletteuse.

    Compared with Pillow per-frame quantization, this
    substantially improves:
        - skin tone
        - dark cabin regions
        - green outdoor areas
        - temporal color consistency
    """

    ffmpeg = check_ffmpeg()

    output_gif.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = max(
        0.0,
        event_time - pre_event,
    )

    duration = (
        pre_event
        + post_event
    )

    print()
    print(
        "[VehicleMind] Generating GitHub GIF..."
    )

    print(
        f"[VehicleMind] GIF window: "
        f"{start_time:.2f}s "
        f"-> "
        f"{start_time + duration:.2f}s"
    )

    print(
        f"[VehicleMind] GIF FPS: "
        f"{gif_fps}"
    )

    print(
        f"[VehicleMind] GIF width: "
        f"{gif_width}px"
    )

    # --------------------------------------------------------
    # High-quality GIF filter
    #
    # Processing:
    #
    # MP4
    #  ↓
    # fps
    #  ↓
    # Lanczos resize
    #  ↓
    # split
    #  ├─ palettegen
    #  └─ original frames
    #         ↓
    #     paletteuse
    #         ↓
    #        GIF
    #
    # One global palette is generated for the selected
    # event window.
    # --------------------------------------------------------

    filter_complex = (
        f"[0:v]"
        f"fps={gif_fps},"
        f"scale={gif_width}:-1:"
        f"flags=lanczos,"
        f"split[s0][s1];"
        f"[s0]"
        f"palettegen="
        f"max_colors=256:"
        f"stats_mode=diff[p];"
        f"[s1][p]"
        f"paletteuse="
        f"dither=sierra2_4a:"
        f"diff_mode=rectangle"
    )

    command = [
        ffmpeg,
        "-y",

        # Input result video
        "-i",
        str(input_video),

        # Accurate event-centered seek
        "-ss",
        f"{start_time:.3f}",

        "-t",
        f"{duration:.3f}",

        "-filter_complex",
        filter_complex,

        "-loop",
        "0",

        str(output_gif),
    ]

    try:

        subprocess.run(
            command,
            check=True,
        )

    except subprocess.CalledProcessError as exc:

        raise RuntimeError(
            "FFmpeg failed while generating GIF."
        ) from exc

    if not output_gif.exists():

        raise RuntimeError(
            "FFmpeg finished but GIF file "
            "was not created."
        )

    gif_size_mb = (
        output_gif.stat().st_size
        / (1024 * 1024)
    )

    print(
        "[VehicleMind] High-quality GIF saved: "
        f"{output_gif}"
    )

    print(
        f"[VehicleMind] GIF size: "
        f"{gif_size_mb:.2f} MB"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Paths
    # ========================================================

    video_path = (
        PROJECT_ROOT
        / args.video
    )

    output_video_path = (
        PROJECT_ROOT
        / args.output_video
    )

    output_gif_path = (
        PROJECT_ROOT
        / args.output_gif
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Demo video not found: "
            f"{video_path}"
        )

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

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Failed to open video: "
            f"{video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:

        source_fps = 30.0

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

    video_duration = (
        total_frames
        / source_fps
        if source_fps > 0
        else 0.0
    )

    print(
        f"[VehicleMind] Input: "
        f"{video_path}"
    )

    print(
        f"[VehicleMind] Resolution: "
        f"{width}x{height}"
    )

    print(
        f"[VehicleMind] FPS: "
        f"{source_fps:.2f}"
    )

    print(
        f"[VehicleMind] Frames: "
        f"{total_frames}"
    )

    print(
        f"[VehicleMind] Duration: "
        f"{video_duration:.2f}s"
    )

    # ========================================================
    # Output video writer
    # ========================================================

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        str(
            output_video_path
        ),
        fourcc,
        source_fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Failed to initialize "
            "output video writer."
        )

    # ========================================================
    # VehicleMind perception modules
    # ========================================================

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

    perclos_estimator = (
        PerclosEstimator(
            window_seconds=30.0,
            min_observation_seconds=5.0,
        )
    )

    # --------------------------------------------------------
    # Driver state thresholds
    #
    # These are demonstration / engineering thresholds.
    #
    # They are not claimed as production DMS,
    # medical, or traffic-safety thresholds.
    # --------------------------------------------------------

    driver_state_estimator = (
        DriverStateEstimator(
            suspected_perclos=0.25,
            drowsy_perclos=0.30,
            suspected_closure_seconds=1.2,
            drowsy_closure_seconds=2.0,
        )
    )

    rest_advisor = (
        MockRestAdvisor()
    )

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

            success, frame = (
                cap.read()
            )

            if not success:
                break

            # ==================================================
            # Offline-video timestamp
            #
            # Temporal fatigue features MUST use video time,
            # rather than CPU / wall-clock processing time.
            # ==================================================

            timestamp_ms = int(
                (
                    frame_index
                    / source_fps
                )
                * 1000
            )

            # MediaPipe VIDEO mode requires monotonically
            # increasing timestamps.
            timestamp_ms = max(
                timestamp_ms,
                previous_timestamp_ms + 1,
            )

            previous_timestamp_ms = (
                timestamp_ms
            )

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

                eye_result = (
                    eye_analyzer.analyze(
                        face,
                        width,
                        height,
                    )
                )

                diag_ear_values.append(
                    eye_result.mean_ear
                )

                # ----------------------------------------------
                # Continuous closure diagnostics
                # ----------------------------------------------

                if eye_result.is_closed:

                    diag_closed_frames += 1

                    if (
                        diag_closed_start_ms
                        is None
                    ):

                        diag_closed_start_ms = (
                            timestamp_ms
                        )

                    current_closure_ms = (
                        timestamp_ms
                        - diag_closed_start_ms
                    )

                    diag_max_closure_ms = max(
                        diag_max_closure_ms,
                        current_closure_ms,
                    )

                else:

                    diag_closed_start_ms = None

                # ----------------------------------------------
                # Blink detection
                # ----------------------------------------------

                blink_detector.update(
                    eye_result.is_closed
                )

                # ----------------------------------------------
                # PERCLOS
                # ----------------------------------------------

                perclos_result = (
                    perclos_estimator.update(
                        timestamp_ms,
                        eye_result.is_closed,
                    )
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

                # ----------------------------------------------
                # State diagnostics
                # ----------------------------------------------

                if (
                    driver_state_result.state
                    == DriverState.SUSPECTED
                ):

                    diag_suspected_frames += 1

                if (
                    driver_state_result.state
                    == DriverState.DROWSY
                ):

                    diag_drowsy_frames += 1

                    recommendation = (
                        rest_advisor.recommend()
                    )

                    # ------------------------------------------
                    # Record the FIRST DROWSY event only.
                    # ------------------------------------------

                    if (
                        first_drowsy_event_time
                        is None
                    ):

                        first_drowsy_event_time = (
                            timestamp_ms
                            / 1000.0
                        )

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

            if (
                driver_state_result
                is not None
            ):

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

            writer.write(
                frame
            )

            # ==================================================
            # Optional preview
            # ==================================================

            if args.show:

                cv2.imshow(
                    "VehicleMind - Public Demo",
                    frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):
                    break

            frame_index += 1

    finally:

        cap.release()

        writer.release()

        detector.close()

        cv2.destroyAllWindows()

    # ========================================================
    # Result MP4
    # ========================================================

    print()

    print(
        "[VehicleMind] "
        "Processed video saved: "
        f"{output_video_path}"
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    print()
    print("=" * 60)
    print(
        "VehicleMind Cabin Diagnostics"
    )
    print("=" * 60)

    print(
        f"Face detection coverage: "
        f"{diag_face_frames}/{frame_index} "
        f"("
        f"{diag_face_frames / max(frame_index, 1) * 100:.1f}%"
        f")"
    )

    if diag_ear_values:

        ear_array = np.asarray(
            diag_ear_values,
            dtype=np.float32,
        )

        percentiles = np.percentile(
            ear_array,
            [
                1,
                5,
                10,
                25,
                50,
                75,
                90,
                95,
                99,
            ],
        )

        print()
        print(
            "EAR statistics"
        )
        print(
            "-" * 40
        )

        print(
            "Configured EAR threshold: "
            f"{eye_analyzer.ear_threshold:.3f}"
        )

        print(
            f"EAR min:    "
            f"{ear_array.min():.3f}"
        )

        print(
            f"EAR mean:   "
            f"{ear_array.mean():.3f}"
        )

        print(
            f"EAR median: "
            f"{np.median(ear_array):.3f}"
        )

        labels = [
            "P01",
            "P05",
            "P10",
            "P25",
            "P50",
            "P75",
            "P90",
            "P95",
            "P99",
        ]

        for label, value in zip(
            labels,
            percentiles,
        ):

            print(
                f"{label}: "
                f"{value:.3f}"
            )

        print()

        print(
            "Frames classified CLOSED: "
            f"{diag_closed_frames}"
        )

        print(
            "Closed-frame ratio: "
            f"{diag_closed_frames / max(diag_face_frames, 1) * 100:.1f}%"
        )

    else:

        print()
        print(
            "No valid EAR observations."
        )

    print()
    print(
        "Temporal statistics"
    )
    print(
        "-" * 40
    )

    print(
        "Maximum continuous closure: "
        f"{diag_max_closure_ms / 1000:.2f}s"
    )

    print(
        "Maximum PERCLOS (all): "
        f"{diag_max_perclos * 100:.1f}%"
    )

    print(
        "Maximum PERCLOS (ready): "
        f"{diag_max_ready_perclos * 100:.1f}%"
    )

    print(
        "SUSPECTED frames: "
        f"{diag_suspected_frames}"
    )

    print(
        "DROWSY frames: "
        f"{diag_drowsy_frames}"
    )

    if (
        first_drowsy_event_time
        is not None
    ):

        print(
            "First DROWSY event: "
            f"{first_drowsy_event_time:.2f}s"
        )

    print(
        "=" * 60
    )

    # ========================================================
    # Generate high-quality GIF with FFmpeg
    # ========================================================

    if (
        first_drowsy_event_time
        is None
    ):

        print()

        print(
            "[VehicleMind] WARNING: "
            "No DROWSY event was detected."
        )

        print(
            "[VehicleMind] "
            "GIF was not generated."
        )

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

    print(
        "[VehicleMind] "
        "Demo generation complete."
    )


if __name__ == "__main__":
    main()
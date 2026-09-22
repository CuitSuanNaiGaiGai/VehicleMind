from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2


from modules.cabin.perception_service import (
    CabinPerceptionService,
)

from modules.driving.perception_service import (
    DrivingPerceptionService,
)


from modules.vehicle_ai.runtime import (
    VehicleMindRuntime,
)


# ============================================================
# Project
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


CABIN_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


DRIVING_MODEL = PROJECT_ROOT / "models" / "driving" / "YOLOPv2_512.onnx"


ROAD_VIDEO = PROJECT_ROOT / "assets" / "driving" / "road_test.mp4"


# ============================================================
# Cabin worker
# ============================================================


def cabin_worker(
    runtime: VehicleMindRuntime,
    stop_event: threading.Event,
):
    """
    Continuously run Cabin Intelligence.

    The worker only updates VehicleMind context.
    It does not call the LLM.
    """

    print("[Cabin] Initializing...")

    service = CabinPerceptionService(model_path=(CABIN_MODEL))

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION,
    )

    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        service.close()

        raise RuntimeError("Cabin camera unavailable.")

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    # --------------------------------------------------------
    # Camera warm-up
    #
    # Avoid treating the very first unstable camera frame
    # as meaningful perception input.
    # --------------------------------------------------------

    for _ in range(5):
        if stop_event.is_set():
            break

        camera.read()

    print("[Cabin] Running.")

    last_log = time.monotonic()

    try:
        while not (stop_event.is_set()):
            ok, frame = camera.read()

            if not ok:
                print("[Cabin] Failed to read frame.")

                time.sleep(0.05)

                continue

            # --------------------------------------------
            # Same behavior as current Cabin Demo.
            # --------------------------------------------

            frame = cv2.flip(
                frame,
                1,
            )

            # --------------------------------------------
            # Perception
            # --------------------------------------------

            snapshot = service.process_frame(frame)

            # --------------------------------------------
            # SAME shared VehicleMind runtime
            # --------------------------------------------

            runtime.update_cabin(**snapshot.to_context_kwargs())

            # --------------------------------------------
            # Lightweight status logging.
            # --------------------------------------------

            now = time.monotonic()

            if now - last_log >= 2.0:
                state_text = getattr(
                    snapshot.driver_state,
                    "value",
                    snapshot.driver_state,
                )

                presence_text = getattr(
                    snapshot.presence,
                    "value",
                    snapshot.presence,
                )

                print(
                    "[Cabin] "
                    f"presence={presence_text} "
                    f"state={state_text} "
                    f"face={snapshot.face_visible}"
                )

                last_log = now

    finally:
        camera.release()

        service.close()

        print("[Cabin] Stopped.")


# ============================================================
# Driving worker
# ============================================================


def driving_worker(
    runtime: VehicleMindRuntime,
    stop_event: threading.Event,
):
    """
    Continuously run Driving Perception using road_test.mp4.

    The video is looped to simulate a persistent front camera.
    """

    print("[Driving] Initializing...")

    service = DrivingPerceptionService(
        model_path=(DRIVING_MODEL),
        work_width=1280,
        work_height=720,
        score_threshold=0.30,
        nms_threshold=0.45,
        prefer_coreml=True,
        warmup_runs=2,
    )

    video = cv2.VideoCapture(str(ROAD_VIDEO))

    if not video.isOpened():
        service.close()

        raise RuntimeError(f"Road video unavailable: {ROAD_VIDEO}")

    source_fps = float(video.get(cv2.CAP_PROP_FPS))

    if source_fps <= 0:
        source_fps = 30.0

    frame_period = 1.0 / source_fps

    print(f"[Driving] Running at {source_fps:.2f} FPS source rate.")

    last_log = time.monotonic()

    try:
        while not (stop_event.is_set()):
            frame_start = time.monotonic()

            ok, frame = video.read()

            # --------------------------------------------
            # Loop road video.
            # --------------------------------------------

            if not ok:
                video.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0,
                )

                continue

            # --------------------------------------------
            # Perception
            # --------------------------------------------

            snapshot = service.process_frame(frame)

            # --------------------------------------------
            # SAME shared VehicleMind runtime
            # --------------------------------------------

            runtime.update_driving(**snapshot.to_context_kwargs())

            # --------------------------------------------
            # Periodic logging
            # --------------------------------------------

            now = time.monotonic()

            if now - last_log >= 2.0:
                print(
                    "[Driving] "
                    f"vehicles="
                    f"{snapshot.vehicle_count} "
                    f"pedestrians="
                    f"{snapshot.pedestrian_count} "
                    f"lane="
                    f"{snapshot.lane_detected} "
                    f"drivable="
                    f"{snapshot.drivable_area_detected}"
                )

                last_log = now

            # --------------------------------------------
            # Preserve approximately the temporal rate of
            # the source video.
            #
            # If inference is already slower than frame
            # budget, no extra sleep is added.
            # --------------------------------------------

            elapsed = time.monotonic() - frame_start

            sleep_seconds = frame_period - elapsed

            if sleep_seconds > 0:
                stop_event.wait(sleep_seconds)

    finally:
        video.release()

        service.close()

        print("[Driving] Stopped.")

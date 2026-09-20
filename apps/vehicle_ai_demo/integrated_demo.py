from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from dotenv import load_dotenv

from modules.cabin.perception_service import (
    CabinPerceptionService,
)

from modules.driving.perception_service import (
    DrivingPerceptionService,
)

from modules.vehicle_ai.context import (
    GearState,
)

from modules.vehicle_ai.llm import (
    build_llm_client,
)

from modules.vehicle_ai.runtime import (
    VehicleMindRuntime,
)


# ============================================================
# Project
# ============================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


CABIN_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)


DRIVING_MODEL = (
    PROJECT_ROOT
    / "models"
    / "driving"
    / "YOLOPv2_512.onnx"
)


ROAD_VIDEO = (
    PROJECT_ROOT
    / "assets"
    / "driving"
    / "road_test.mp4"
)


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

    print(
        "[Cabin] Initializing..."
    )

    service = (
        CabinPerceptionService(
            model_path=(
                CABIN_MODEL
            )
        )
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION,
    )

    if not camera.isOpened():

        camera = (
            cv2.VideoCapture(
                0
            )
        )

    if not camera.isOpened():

        service.close()

        raise RuntimeError(
            "Cabin camera unavailable."
        )

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

    print(
        "[Cabin] Running."
    )

    last_log = (
        time.monotonic()
    )

    try:

        while not (
            stop_event.is_set()
        ):

            ok, frame = (
                camera.read()
            )

            if not ok:

                print(
                    "[Cabin] "
                    "Failed to read frame."
                )

                time.sleep(
                    0.05
                )

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

            snapshot = (
                service.process_frame(
                    frame
                )
            )

            # --------------------------------------------
            # SAME shared VehicleMind runtime
            # --------------------------------------------

            runtime.update_cabin(
                **snapshot
                .to_context_kwargs()
            )

            # --------------------------------------------
            # Lightweight status logging.
            # --------------------------------------------

            now = (
                time.monotonic()
            )

            if (
                now
                - last_log
                >= 2.0
            ):

                state_text = (
                    getattr(
                        snapshot.driver_state,
                        "value",
                        snapshot.driver_state,
                    )
                )

                presence_text = (
                    getattr(
                        snapshot.presence,
                        "value",
                        snapshot.presence,
                    )
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

        print(
            "[Cabin] Stopped."
        )


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

    print(
        "[Driving] Initializing..."
    )

    service = (
        DrivingPerceptionService(
            model_path=(
                DRIVING_MODEL
            ),
            work_width=1280,
            work_height=720,
            score_threshold=0.30,
            nms_threshold=0.45,
            prefer_coreml=True,
            warmup_runs=2,
        )
    )

    video = cv2.VideoCapture(
        str(
            ROAD_VIDEO
        )
    )

    if not video.isOpened():

        service.close()

        raise RuntimeError(
            "Road video unavailable: "
            f"{ROAD_VIDEO}"
        )

    source_fps = float(
        video.get(
            cv2.CAP_PROP_FPS
        )
    )

    if source_fps <= 0:

        source_fps = 30.0

    frame_period = (
        1.0
        / source_fps
    )

    print(
        "[Driving] Running at "
        f"{source_fps:.2f} FPS source rate."
    )

    last_log = (
        time.monotonic()
    )

    try:

        while not (
            stop_event.is_set()
        ):

            frame_start = (
                time.monotonic()
            )

            ok, frame = (
                video.read()
            )

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

            snapshot = (
                service.process_frame(
                    frame
                )
            )

            # --------------------------------------------
            # SAME shared VehicleMind runtime
            # --------------------------------------------

            runtime.update_driving(
                **snapshot
                .to_context_kwargs()
            )

            # --------------------------------------------
            # Periodic logging
            # --------------------------------------------

            now = (
                time.monotonic()
            )

            if (
                now
                - last_log
                >= 2.0
            ):

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

            elapsed = (
                time.monotonic()
                - frame_start
            )

            sleep_seconds = (
                frame_period
                - elapsed
            )

            if sleep_seconds > 0:

                stop_event.wait(
                    sleep_seconds
                )

    finally:

        video.release()

        service.close()

        print(
            "[Driving] Stopped."
        )


# ============================================================
# Main
# ============================================================


def main():

    load_dotenv()

    print()
    print(
        "========================================"
    )

    print(
        " VehicleMind Integrated Vehicle AI"
    )

    print(
        " Cabin + Driving + Context + LLM"
    )

    print(
        "========================================"
    )

    # ========================================================
    # LLM
    # ========================================================

    llm = (
        build_llm_client()
    )

    # ========================================================
    # ONE runtime
    # ONE ContextManager
    # ONE Agent
    # ========================================================

    runtime = (
        VehicleMindRuntime(
            llm=llm
        )
    )

    # ========================================================
    # Mock vehicle state
    #
    # Later this can be replaced by Android Vehicle API /
    # CAN / simulator.
    # ========================================================

    runtime.context_manager.update_vehicle(
        speed_kmh=68.0,
        gear=GearState.D,
        cabin_temperature_c=28.0,
        target_temperature_c=24.0,
        ac_enabled=True,
        volume=25,
    )

    # ========================================================
    # Stop signal shared by workers
    # ========================================================

    stop_event = (
        threading.Event()
    )

    # ========================================================
    # Workers
    # ========================================================

    cabin_thread = (
        threading.Thread(
            target=cabin_worker,
            args=(
                runtime,
                stop_event,
            ),
            name=(
                "CabinPerception"
            ),
            daemon=True,
        )
    )

    driving_thread = (
        threading.Thread(
            target=driving_worker,
            args=(
                runtime,
                stop_event,
            ),
            name=(
                "DrivingPerception"
            ),
            daemon=True,
        )
    )

    cabin_thread.start()

    driving_thread.start()

    # ========================================================
    # Give perception a short startup window
    # ========================================================

    print()
    print(
        "[VehicleMind] "
        "Waiting for perception..."
    )

    time.sleep(
        2.0
    )

    print()
    print(
        "[VehicleMind] Ready."
    )

    print(
        "Commands:"
    )

    print(
        "  context  - current unified context"
    )

    print(
        "  fresh    - perception freshness"
    )

    print(
        "  events   - recent semantic events"
    )

    print(
        "  quit     - exit"
    )

    # ========================================================
    # Agent loop
    #
    # LLM runs ONLY when the user speaks.
    # Perception continues independently.
    # ========================================================

    try:

        while True:

            print()

            try:

                text = input(
                    "You > "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):

                print()
                break

            if not text:

                continue

            command = (
                text.lower()
            )

            # --------------------------------------------
            # Quit
            # --------------------------------------------

            if command in {
                "q",
                "quit",
                "exit",
            }:

                break

            # --------------------------------------------
            # Context
            # --------------------------------------------

            if command == "context":

                print(
                    runtime
                    .context_summary()
                )

                continue

            # --------------------------------------------
            # Freshness
            # --------------------------------------------

            if command == "fresh":

                freshness = (
                    runtime
                    .context_manager
                    .freshness()
                )

                print()

                for domain, info in (
                    freshness.items()
                ):

                    print(
                        f"{domain:<8} "
                        f"age="
                        f"{info['age_seconds']:.3f}s "
                        f"fresh="
                        f"{info['fresh']}"
                    )

                continue

            # --------------------------------------------
            # Events
            # --------------------------------------------

            if command == "events":

                events = (
                    runtime
                    .event_bus
                    .recent_events(
                        limit=20
                    )
                )

                if not events:

                    print(
                        "No events."
                    )

                else:

                    for event in events:

                        print(
                            event
                        )

                continue

            # --------------------------------------------
            # Vehicle Agent
            # --------------------------------------------

            try:

                answer = (
                    runtime.chat(
                        text,
                        debug=True,
                    )
                )

            except Exception as exc:

                print()
                print(
                    "[VehicleMind ERROR]"
                )

                print(
                    type(exc).__name__,
                    str(exc),
                )

                continue

            print()
            print(
                "VehicleMind >",
                answer,
            )

    finally:

        # ====================================================
        # Graceful shutdown
        # ====================================================

        print()
        print(
            "[VehicleMind] "
            "Stopping perception..."
        )

        stop_event.set()

        cabin_thread.join(
            timeout=5.0
        )

        driving_thread.join(
            timeout=5.0
        )

        print(
            "[VehicleMind] Shutdown complete."
        )


if __name__ == "__main__":
    main()
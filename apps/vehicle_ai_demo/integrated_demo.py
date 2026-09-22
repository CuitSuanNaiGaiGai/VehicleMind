from __future__ import annotations

import threading
import time
from pathlib import Path


from dotenv import load_dotenv

from apps.vehicle_ai_demo.integrated_workers import cabin_worker, driving_worker

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


CABIN_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


DRIVING_MODEL = PROJECT_ROOT / "models" / "driving" / "YOLOPv2_512.onnx"


ROAD_VIDEO = PROJECT_ROOT / "assets" / "driving" / "road_test.mp4"


def main():

    load_dotenv()

    print()
    print("========================================")

    print(" VehicleMind Integrated Vehicle AI")

    print(" Cabin + Driving + Context + LLM")

    print("========================================")

    # ========================================================
    # LLM
    # ========================================================

    llm = build_llm_client()

    # ========================================================
    # ONE runtime
    # ONE ContextManager
    # ONE Agent
    # ========================================================

    runtime = VehicleMindRuntime(llm=llm)

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

    stop_event = threading.Event()

    # ========================================================
    # Workers
    # ========================================================

    cabin_thread = threading.Thread(
        target=cabin_worker,
        args=(
            runtime,
            stop_event,
        ),
        name=("CabinPerception"),
        daemon=True,
    )

    driving_thread = threading.Thread(
        target=driving_worker,
        args=(
            runtime,
            stop_event,
        ),
        name=("DrivingPerception"),
        daemon=True,
    )

    cabin_thread.start()

    driving_thread.start()

    # ========================================================
    # Give perception a short startup window
    # ========================================================

    print()
    print("[VehicleMind] Waiting for perception...")

    time.sleep(2.0)

    print()
    print("[VehicleMind] Ready.")

    print("Commands:")

    print("  context  - current unified context")

    print("  fresh    - perception freshness")

    print("  events   - recent semantic events")

    print("  quit     - exit")

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
                text = input("You > ").strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):
                print()
                break

            if not text:
                continue

            command = text.lower()

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
                print(runtime.context_summary())

                continue

            # --------------------------------------------
            # Freshness
            # --------------------------------------------

            if command == "fresh":
                freshness = runtime.context_manager.freshness()

                print()

                for domain, info in freshness.items():
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
                events = runtime.event_bus.recent_events(limit=20)

                if not events:
                    print("No events.")

                else:
                    for event in events:
                        print(event)

                continue

            # --------------------------------------------
            # Vehicle Agent
            # --------------------------------------------

            try:
                answer = runtime.chat(
                    text,
                    debug=True,
                )

            except Exception as exc:
                print()
                print("[VehicleMind ERROR]")

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
        print("[VehicleMind] Stopping perception...")

        stop_event.set()

        cabin_thread.join(timeout=5.0)

        driving_thread.join(timeout=5.0)

        print("[VehicleMind] Shutdown complete.")


if __name__ == "__main__":
    main()

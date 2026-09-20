from dotenv import load_dotenv

from modules.vehicle_ai.context import (
    GearState,
)

from modules.vehicle_ai.llm import (
    build_llm_client,
)

from modules.vehicle_ai.runtime import (
    VehicleMindRuntime,
)


def main():

    load_dotenv()

    # ========================================================
    # LLM
    # ========================================================

    llm = (
        build_llm_client()
    )

    # ========================================================
    # ONE unified VehicleMind runtime
    # ========================================================

    runtime = (
        VehicleMindRuntime(
            llm=llm
        )
    )

    # ========================================================
    # Vehicle State
    #
    # Later comes from vehicle / simulator / Android API.
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
    # Cabin Perception output
    #
    # Simulates the actual Cabin pipeline.
    # ========================================================

    runtime.update_cabin(
        face_present=True,
        driver_state="DROWSY",
        risk="HIGH",
        perclos=0.31,
        eye_closed=True,
        eye_closure_seconds=2.2,
        recent_yawns=2,
        blink_count=18,
    )

    # ========================================================
    # Driving Perception output
    #
    # Simulates the actual Driving pipeline.
    # ========================================================

    runtime.update_driving(
        vehicle_count=14,
        pedestrian_count=2,
        rider_count=1,
        traffic_light_count=2,
        traffic_sign_count=3,
        lane_detected=True,
        drivable_area_detected=True,
    )

    # ========================================================
    # Inspect unified context
    # ========================================================

    print()
    print(
        runtime.context_summary()
    )

    print(
        "VehicleMind Integrated Runtime Ready"
    )

    print(
        "Type 'context' to inspect context."
    )

    print(
        "Type 'quit' to exit."
    )

    # ========================================================
    # Agent CLI
    # ========================================================

    while True:

        print()

        try:

            text = input(
                "You > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if not text:

            continue

        if (
            text.lower()
            in {
                "quit",
                "exit",
                "q",
            }
        ):

            break

        if (
            text.lower()
            == "context"
        ):

            print(
                runtime
                .context_summary()
            )

            continue

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


if __name__ == "__main__":
    main()
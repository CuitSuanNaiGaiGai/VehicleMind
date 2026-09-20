from dotenv import load_dotenv

from modules.vehicle_ai.agent import (
    VehicleAgent,
)

from modules.vehicle_ai.context import (
    ContextManager,
    DriverPresence,
    DriverState,
    GearState,
    RiskLevel,
)

from modules.vehicle_ai.llm import (
    build_llm_client,
)

from modules.vehicle_ai.tools import (
    build_default_tool_registry,
)


def main():

    # ========================================================
    # Environment
    # ========================================================

    load_dotenv()

    print()
    print(
        "========================================"
    )

    print(
        " VehicleMind Vehicle AI"
    )

    print(
        " Context-Aware Vehicle Agent"
    )

    print(
        "========================================"
    )

    # ========================================================
    # Vehicle Context
    # ========================================================

    context_manager = (
        ContextManager()
    )

    # --------------------------------------------------------
    # Mock current vehicle state
    # --------------------------------------------------------

    context_manager.update_vehicle(
        speed_kmh=68.0,
        gear=GearState.D,
        cabin_temperature_c=28.0,
        target_temperature_c=24.0,
        ac_enabled=True,
        volume=25,
    )

    # --------------------------------------------------------
    # Mock Cabin Intelligence
    # --------------------------------------------------------

    context_manager.update_driver(
        presence=(
            DriverPresence.PRESENT
        ),
        state=(
            DriverState.NORMAL
        ),
        risk=(
            RiskLevel.LOW
        ),
        perclos=0.12,
        eye_closure_seconds=0.0,
        recent_yawns=0,
    )

    # --------------------------------------------------------
    # Mock Driving Perception
    # --------------------------------------------------------

    context_manager.update_road(
        vehicle_count=6,
        pedestrian_count=1,
        rider_count=0,
        traffic_light_count=2,
        traffic_sign_count=1,
        total_objects=10,
        lane_detected=True,
        drivable_area_detected=True,
        traffic_level="MODERATE",
    )

    # ========================================================
    # Tools
    # ========================================================

    registry = (
        build_default_tool_registry(
            context_manager
        )
    )

    # ========================================================
    # LLM
    # ========================================================

    llm = (
        build_llm_client()
    )

    # ========================================================
    # Agent
    # ========================================================

    agent = (
        VehicleAgent(
            llm=llm,
            context_manager=(
                context_manager
            ),
            tool_registry=(
                registry
            ),
        )
    )

    # ========================================================
    # CLI
    # ========================================================

    print()
    print(
        "VehicleMind is ready."
    )

    print(
        "Type 'context' to inspect "
        "vehicle context."
    )

    print(
        "Type 'quit' to exit."
    )

    while True:

        print()

        try:

            user_text = input(
                "You > "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if not user_text:

            continue

        if (
            user_text.lower()
            in {
                "quit",
                "exit",
                "q",
            }
        ):

            break

        if (
            user_text.lower()
            == "context"
        ):

            print(
                context_manager
                .summary()
            )

            continue

        try:

            answer = (
                agent.chat(
                    user_text,
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
import json

from modules.vehicle_ai.context import (
    ContextManager,
    DriverPresence,
    DriverState,
    GearState,
    RiskLevel,
)

from modules.vehicle_ai.tools import (
    build_default_tool_registry,
)


# ============================================================
# Helper
# ============================================================


def execute(
    registry,
    name,
    arguments=None,
):
    print()
    print(
        "----------------------------------------"
    )

    print(
        f"[TOOL CALL]"
    )

    print(
        f"  name      : {name}"
    )

    print(
        f"  arguments : "
        f"{arguments or {}}"
    )

    result = (
        registry.execute(
            name=name,
            arguments=arguments,
        )
    )

    print(
        "[TOOL RESULT]"
    )

    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    return result


# ============================================================
# Main
# ============================================================


def main():
    print()
    print(
        "========================================"
    )

    print(
        " VehicleMind Tool Demo"
    )

    print(
        "========================================"
    )

    # ========================================================
    # Context
    # ========================================================

    manager = (
        ContextManager()
    )

    manager.update_vehicle(
        gear=GearState.D,
        speed_kmh=68.0,
        cabin_temperature_c=27.5,
        target_temperature_c=24.0,
        ac_enabled=True,
        volume=28,
    )

    manager.update_driver(
        presence=(
            DriverPresence.PRESENT
        ),
        state=(
            DriverState.DROWSY
        ),
        risk=(
            RiskLevel.HIGH
        ),
        perclos=0.31,
        recent_yawns=2,
    )

    # ========================================================
    # Registry
    # ========================================================

    registry = (
        build_default_tool_registry(
            manager
        )
    )

    print()
    print(
        "Available tools:"
    )

    for name in (
        registry.names()
    ):

        print(
            f"  - {name}"
        )

    # ========================================================
    # Scenario A
    #
    # User: 有点热
    # ========================================================

    execute(
        registry,
        "get_climate_status",
    )

    execute(
        registry,
        "set_temperature",
        {
            "temperature_c": 23,
        },
    )

    # ========================================================
    # Scenario B
    #
    # User: 我有点困，找个地方休息
    # ========================================================

    rest_result = (
        execute(
            registry,
            "search_nearby_rest_area",
        )
    )

    if rest_result.success:

        poi_id = (
            rest_result
            .data[
                "poi_id"
            ]
        )

        execute(
            registry,
            "start_navigation",
            {
                "poi_id":
                    poi_id,
            },
        )

    # ========================================================
    # Scenario C
    #
    # Media control
    # ========================================================

    execute(
        registry,
        "play_music",
        {
            "query":
                "Relaxing Driving Playlist",
        },
    )

    execute(
        registry,
        "set_volume",
        {
            "volume": 20,
        },
    )

    # ========================================================
    # Scenario D
    #
    # Window control
    # ========================================================

    execute(
        registry,
        "set_driver_window",
        {
            "open": True,
        },
    )

    # ========================================================
    # Error handling
    # ========================================================

    execute(
        registry,
        "set_temperature",
        {
            "temperature_c": 50,
        },
    )

    execute(
        registry,
        "does_not_exist",
        {},
    )

    # ========================================================
    # Final Context
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        " Final Vehicle Context"
    )

    print(
        "========================================"
    )

    print(
        manager.summary()
    )

    # ========================================================
    # LLM schemas
    # ========================================================

    print(
        "========================================"
    )

    print(
        " Function Calling Schemas"
    )

    print(
        "========================================"
    )

    print(
        json.dumps(
            registry.llm_schemas(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

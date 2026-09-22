from modules.vehicle_ai.context.models import (
    DriverContext,
    DriverPresence,
    DriverState,
    GearState,
    NavigationState,
    RiskLevel,
    RoadContext,
    VehicleContext,
    VehicleStatus,
)


def main():
    # ========================================================
    # Mock Cabin Intelligence output
    # ========================================================

    driver = DriverContext(
        presence=(DriverPresence.PRESENT),
        state=(DriverState.SUSPECTED),
        risk=(RiskLevel.MEDIUM),
        perclos=0.27,
        eye_closed=False,
        eye_closure_seconds=0.35,
        recent_yawns=2,
        blink_count=18,
    )

    # ========================================================
    # Mock Driving Perception output
    # ========================================================

    road = RoadContext(
        vehicle_count=6,
        pedestrian_count=1,
        rider_count=1,
        traffic_light_count=2,
        traffic_sign_count=3,
        total_objects=13,
        lane_detected=True,
        drivable_area_detected=True,
        traffic_level="MODERATE",
    )

    # ========================================================
    # Mock vehicle status
    # ========================================================

    vehicle = VehicleStatus(
        speed_kmh=68.0,
        gear=(GearState.D),
        cabin_temperature_c=27.5,
        target_temperature_c=24.0,
        ac_enabled=True,
        driver_window_open=False,
        passenger_window_open=False,
        media_playing=True,
        media_title="Driving Playlist",
        volume=28,
        navigation_state=(NavigationState.IDLE),
    )

    # ========================================================
    # Unified context
    # ========================================================

    context = VehicleContext(
        driver=driver,
        road=road,
        vehicle=vehicle,
    )

    # ========================================================
    # Human-readable
    # ========================================================

    print(context.summary())

    # ========================================================
    # Agent-facing JSON
    # ========================================================

    print("Agent Context:")

    print(context.to_agent_json())

    # ========================================================
    # Freshness
    # ========================================================

    print()
    print("Freshness:")

    print("  Driver:", driver.is_fresh())

    print("  Road:", road.is_fresh())

    print("  Vehicle:", vehicle.is_fresh())


if __name__ == "__main__":
    main()

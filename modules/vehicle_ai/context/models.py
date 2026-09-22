from __future__ import annotations

import json
import time

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from modules.vehicle_ai.context.contract import CONTEXT_SCHEMA_VERSION
from modules.vehicle_ai.context.enums import (
    DriverPresence,
    DriverState,
    GearState,
    NavigationState,
    RiskLevel,
)

# ============================================================
# Utility
# ============================================================


def current_timestamp() -> float:
    """
    Wall-clock timestamp.

    This timestamp represents when a context snapshot was
    produced or last updated.

    We intentionally use Unix timestamp here because context
    may later be serialized, logged, replayed or passed
    between processes.
    """

    return time.time()


# ============================================================
# Driver Context
# ============================================================


@dataclass
class DriverContext:
    """
    High-level information produced by Cabin Intelligence.

    The future LLM should consume this structure instead of
    directly consuming low-level MediaPipe landmarks.
    """

    presence: DriverPresence = DriverPresence.UNKNOWN

    state: DriverState = DriverState.UNKNOWN

    risk: RiskLevel = RiskLevel.UNKNOWN

    # --------------------------------------------------------
    # Fatigue evidence
    # --------------------------------------------------------

    perclos: float | None = None

    eye_closed: bool | None = None

    eye_closure_seconds: float = 0.0

    recent_yawns: int = 0

    blink_count: int = 0

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    updated_at: float = field(default_factory=current_timestamp)

    source: str = "cabin_perception"

    def touch(self) -> None:
        """
        Mark this context as freshly updated.
        """

        self.updated_at = current_timestamp()

    def age_seconds(
        self,
        now: float | None = None,
    ) -> float:

        if now is None:
            now = current_timestamp()

        return max(
            0.0,
            now - self.updated_at,
        )

    def is_fresh(
        self,
        max_age_seconds: float = 2.0,
        now: float | None = None,
    ) -> bool:

        return self.age_seconds(now) <= max_age_seconds


# ============================================================
# Road Context
# ============================================================


@dataclass
class RoadContext:
    """
    High-level road-scene information produced by
    Driving Perception.

    Notice that the agent receives semantic information,
    not lane masks or raw bounding boxes.
    """

    vehicle_count: int = 0

    pedestrian_count: int = 0

    rider_count: int = 0

    traffic_light_count: int = 0

    traffic_sign_count: int = 0

    total_objects: int = 0

    lane_detected: bool = False

    drivable_area_detected: bool = False

    # --------------------------------------------------------
    # Optional compact scene description
    # --------------------------------------------------------

    traffic_level: str = "UNKNOWN"

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    updated_at: float = field(default_factory=current_timestamp)

    source: str = "driving_perception"

    def touch(self) -> None:

        self.updated_at = current_timestamp()

    def age_seconds(
        self,
        now: float | None = None,
    ) -> float:

        if now is None:
            now = current_timestamp()

        return max(
            0.0,
            now - self.updated_at,
        )

    def is_fresh(
        self,
        max_age_seconds: float = 1.0,
        now: float | None = None,
    ) -> bool:

        return self.age_seconds(now) <= max_age_seconds


# ============================================================
# Vehicle Status
# ============================================================


@dataclass
class VehicleStatus:
    """
    Vehicle-side status.

    At the moment all values are mock / simulated.

    Later this layer may be connected to:
        CAN
        Android Vehicle API
        CARLA
        vehicle middleware
        other platform APIs
    """

    speed_kmh: float = 0.0

    gear: GearState = GearState.P

    # --------------------------------------------------------
    # Climate
    # --------------------------------------------------------

    cabin_temperature_c: float = 24.0

    target_temperature_c: float = 24.0

    ac_enabled: bool = False

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    driver_window_open: bool = False

    passenger_window_open: bool = False

    # --------------------------------------------------------
    # Media
    # --------------------------------------------------------

    media_playing: bool = False

    media_title: str | None = None

    volume: int = 30

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    navigation_state: NavigationState = NavigationState.IDLE
    navigation_destination_id: str | None = None
    navigation_destination: str | None = None

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    updated_at: float = field(default_factory=current_timestamp)

    source: str = "vehicle_state"

    def touch(self) -> None:

        self.updated_at = current_timestamp()

    def age_seconds(
        self,
        now: float | None = None,
    ) -> float:

        if now is None:
            now = current_timestamp()

        return max(
            0.0,
            now - self.updated_at,
        )

    def is_fresh(
        self,
        max_age_seconds: float = 2.0,
        now: float | None = None,
    ) -> bool:

        return self.age_seconds(now) <= max_age_seconds


# ============================================================
# Vehicle Context
# ============================================================


@dataclass
class VehicleContext:
    """
    Unified VehicleMind context snapshot.

    This is the central data structure used by the future:

        ContextManager
        ContextSelector
        EventBus
        VehicleAgent
        SafetyPolicy
        Tool system

    The LLM should primarily consume this layer instead of
    directly accessing individual perception modules.
    """

    schema_version: int = field(default=CONTEXT_SCHEMA_VERSION, init=False)

    driver: DriverContext = field(default_factory=DriverContext)

    road: RoadContext = field(default_factory=RoadContext)

    vehicle: VehicleStatus = field(default_factory=VehicleStatus)

    created_at: float = field(default_factory=current_timestamp)

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Full machine-readable representation.
        """

        return asdict(self)

    def to_json(
        self,
        indent: int = 2,
    ) -> str:
        """
        JSON representation useful for debugging and logs.
        """

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
        )

    # ========================================================
    # LLM-facing representation
    # ========================================================

    def to_agent_context(
        self,
    ) -> dict[str, Any]:
        """
        Return a compact semantic representation.

        Internal metadata such as timestamps and source names
        are intentionally excluded here.

        ContextSelector will later further reduce this structure
        according to the user's current task.
        """

        return {
            "driver": {
                "presence": (self.driver.presence),
                "state": (self.driver.state),
                "risk": (self.driver.risk),
                "perclos": (self.driver.perclos),
                "eye_closed": (self.driver.eye_closed),
                "eye_closure_seconds": (self.driver.eye_closure_seconds),
                "recent_yawns": (self.driver.recent_yawns),
            },
            "road": {
                "vehicle_count": (self.road.vehicle_count),
                "pedestrian_count": (self.road.pedestrian_count),
                "rider_count": (self.road.rider_count),
                "traffic_light_count": (self.road.traffic_light_count),
                "traffic_sign_count": (self.road.traffic_sign_count),
                "total_objects": (self.road.total_objects),
                "lane_detected": (self.road.lane_detected),
                "drivable_area_detected": (self.road.drivable_area_detected),
                "traffic_level": (self.road.traffic_level),
            },
            "vehicle": {
                "speed_kmh": (self.vehicle.speed_kmh),
                "gear": (self.vehicle.gear),
                "cabin_temperature_c": (self.vehicle.cabin_temperature_c),
                "target_temperature_c": (self.vehicle.target_temperature_c),
                "ac_enabled": (self.vehicle.ac_enabled),
                "driver_window_open": (self.vehicle.driver_window_open),
                "media_playing": (self.vehicle.media_playing),
                "media_title": (self.vehicle.media_title),
                "volume": (self.vehicle.volume),
                "navigation_state": (self.vehicle.navigation_state),
                "navigation_destination_id": (self.vehicle.navigation_destination_id),
                "navigation_destination": (self.vehicle.navigation_destination),
            },
        }

    def to_agent_json(
        self,
        indent: int = 2,
    ) -> str:
        """
        Compact JSON representation intended for future
        LLM context injection.
        """

        return json.dumps(
            self.to_agent_context(),
            ensure_ascii=False,
            indent=indent,
        )

    # ========================================================
    # Human-readable summary
    # ========================================================

    def summary(self) -> str:
        """
        Compact terminal representation.
        """

        perclos_text = (
            "N/A" if self.driver.perclos is None else f"{self.driver.perclos:.3f}"
        )

        return (
            "\n"
            "========== VehicleMind Context ==========\n"
            "\n"
            "DRIVER\n"
            f"  Presence       : "
            f"{self.driver.presence}\n"
            f"  State          : "
            f"{self.driver.state}\n"
            f"  Risk           : "
            f"{self.driver.risk}\n"
            f"  PERCLOS        : "
            f"{perclos_text}\n"
            f"  Eye Closure    : "
            f"{self.driver.eye_closure_seconds:.2f} s\n"
            f"  Recent Yawns   : "
            f"{self.driver.recent_yawns}\n"
            "\n"
            "ROAD\n"
            f"  Vehicles       : "
            f"{self.road.vehicle_count}\n"
            f"  Pedestrians    : "
            f"{self.road.pedestrian_count}\n"
            f"  Riders         : "
            f"{self.road.rider_count}\n"
            f"  Lane           : "
            f"{self.road.lane_detected}\n"
            f"  Drivable Area  : "
            f"{self.road.drivable_area_detected}\n"
            f"  Traffic Level  : "
            f"{self.road.traffic_level}\n"
            "\n"
            "VEHICLE\n"
            f"  Speed          : "
            f"{self.vehicle.speed_kmh:.1f} km/h\n"
            f"  Gear           : "
            f"{self.vehicle.gear}\n"
            f"  Cabin Temp     : "
            f"{self.vehicle.cabin_temperature_c:.1f} C\n"
            f"  Target Temp    : "
            f"{self.vehicle.target_temperature_c:.1f} C\n"
            f"  AC             : "
            f"{self.vehicle.ac_enabled}\n"
            f"  Navigation     : "
            f"{self.vehicle.navigation_state}\n"
            "\n"
            "=========================================\n"
        )

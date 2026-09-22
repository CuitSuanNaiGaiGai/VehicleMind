from __future__ import annotations

from modules.vehicle_ai.context import (
    ContextManager,
)


class DrivingContextAdapter:
    """
    Bridge Driving Perception semantic outputs into RoadContext.

    Important:
        The adapter receives semantic results instead of raw
        YOLO tensors, masks or ONNX outputs.

    Driving perception remains responsible for perception.
    Vehicle AI only consumes the high-level scene state.
    """

    def __init__(
        self,
        context_manager: ContextManager,
    ):
        self.context_manager = context_manager

    @staticmethod
    def estimate_traffic_level(
        vehicle_count: int,
    ) -> str:
        """
        Temporary demo-level heuristic.

        Later this can be replaced with a temporal traffic
        estimator without changing the Agent interface.
        """

        if vehicle_count <= 3:
            return "LIGHT"

        if vehicle_count <= 9:
            return "MODERATE"

        return "HEAVY"

    def update(
        self,
        *,
        vehicle_count: int = 0,
        pedestrian_count: int = 0,
        rider_count: int = 0,
        traffic_light_count: int = 0,
        traffic_sign_count: int = 0,
        lane_detected: bool = False,
        drivable_area_detected: bool = False,
        traffic_level: str | None = None,
    ):
        vehicle_count = int(vehicle_count)

        if traffic_level is None:
            traffic_level = self.estimate_traffic_level(vehicle_count)

        total_objects = (
            int(vehicle_count)
            + int(pedestrian_count)
            + int(rider_count)
            + int(traffic_light_count)
            + int(traffic_sign_count)
        )

        return self.context_manager.update_road(
            vehicle_count=(vehicle_count),
            pedestrian_count=int(pedestrian_count),
            rider_count=int(rider_count),
            traffic_light_count=int(traffic_light_count),
            traffic_sign_count=int(traffic_sign_count),
            total_objects=(total_objects),
            lane_detected=bool(lane_detected),
            drivable_area_detected=bool(drivable_area_detected),
            traffic_level=(traffic_level),
        )

from __future__ import annotations

import time

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from modules.driving.perception.types import DrivingSceneResult
from modules.observation import ObservationMetadata, ObservationSequencer


# ============================================================
# Result
# ============================================================


@dataclass
class DrivingPerceptionSnapshot:
    """
    High-level semantic representation of one road frame.
    """

    metadata: ObservationMetadata

    scene_result: DrivingSceneResult

    working_frame: np.ndarray

    vehicle_count: int

    pedestrian_count: int

    rider_count: int

    traffic_light_count: int

    traffic_sign_count: int

    total_objects: int

    lane_detected: bool

    drivable_area_detected: bool

    lane_pixels: int

    drivable_pixels: int

    drivable_ratio: float

    def to_context_kwargs(
        self,
    ) -> dict:

        return {
            "vehicle_count": self.vehicle_count,
            "pedestrian_count": self.pedestrian_count,
            "rider_count": self.rider_count,
            "traffic_light_count": self.traffic_light_count,
            "traffic_sign_count": self.traffic_sign_count,
            "lane_detected": self.lane_detected,
            "drivable_area_detected": self.drivable_area_detected,
        }


# ============================================================
# Service
# ============================================================


class DrivingPerceptionService:
    """
    Reusable YOLOPv2 driving-perception pipeline.

    One model produces:

        object detection
        drivable-area segmentation
        lane segmentation
    """

    def __init__(
        self,
        model_path: str | Path,
        work_width: int = 1280,
        work_height: int = 720,
        score_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        prefer_coreml: bool = True,
        warmup_runs: int = 2,
    ):
        from modules.driving.perception.panoptic_detector import PanopticDrivingDetector

        self.work_width = int(work_width)

        self.work_height = int(work_height)

        self.detector = PanopticDrivingDetector(
            model_path=Path(model_path),
            score_threshold=(score_threshold),
            nms_threshold=(nms_threshold),
            prefer_coreml=(prefer_coreml),
            warmup_runs=(warmup_runs),
        )
        self._started_at = time.perf_counter()
        self._observations = ObservationSequencer("driving_perception")

    # ========================================================
    # Process one frame
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> DrivingPerceptionSnapshot:
        processing_started = time.perf_counter()
        if timestamp_ms is None:
            timestamp_ms = max(0, int((processing_started - self._started_at) * 1000))

        # ----------------------------------------------------
        # Resize once, consistent with current scene_demo.py
        # ----------------------------------------------------

        if frame.shape[1] != self.work_width or frame.shape[0] != self.work_height:
            working_frame = cv2.resize(
                frame,
                (
                    self.work_width,
                    self.work_height,
                ),
                interpolation=(cv2.INTER_AREA),
            )

        else:
            working_frame = frame.copy()

        # ====================================================
        # YOLOPv2
        # ====================================================

        scene_result = self.detector.detect(working_frame)

        # ====================================================
        # Object counts
        # ====================================================

        counts: dict[
            str,
            int,
        ] = {}

        for obj in scene_result.objects:
            class_name = obj.class_name

            counts[class_name] = (
                counts.get(
                    class_name,
                    0,
                )
                + 1
            )

        # ----------------------------------------------------
        # Motor vehicles
        # ----------------------------------------------------

        vehicle_count = (
            counts.get(
                "car",
                0,
            )
            + counts.get(
                "truck",
                0,
            )
            + counts.get(
                "bus",
                0,
            )
            + counts.get(
                "train",
                0,
            )
        )

        # ----------------------------------------------------
        # Pedestrians
        # ----------------------------------------------------

        pedestrian_count = counts.get(
            "person",
            0,
        )

        # ----------------------------------------------------
        # Two-wheel / rider road users
        #
        # RoadContext currently has one rider_count field,
        # therefore rider + motorcycle + bicycle are grouped
        # here.
        # ----------------------------------------------------

        rider_count = (
            counts.get(
                "rider",
                0,
            )
            + counts.get(
                "motorcycle",
                0,
            )
            + counts.get(
                "bicycle",
                0,
            )
        )

        traffic_light_count = counts.get(
            "traffic light",
            0,
        )

        traffic_sign_count = counts.get(
            "traffic sign",
            0,
        )

        total_objects = len(scene_result.objects)

        # ====================================================
        # Drivable Area
        # ====================================================

        drivable_pixels = int(np.count_nonzero(scene_result.drivable_mask))

        total_pixels = int(scene_result.drivable_mask.size)

        drivable_ratio = drivable_pixels / max(
            total_pixels,
            1,
        )

        drivable_area_detected = drivable_pixels > 1000

        # ====================================================
        # Lane
        # ====================================================

        lane_pixels = int(np.count_nonzero(scene_result.lane_mask))

        lane_detected = lane_pixels > 300

        # ====================================================
        # Snapshot
        # ====================================================

        return DrivingPerceptionSnapshot(
            metadata=self._observations.next(
                timestamp_ms=timestamp_ms,
                processing_ms=(time.perf_counter() - processing_started) * 1000,
            ),
            scene_result=(scene_result),
            working_frame=(working_frame),
            vehicle_count=(vehicle_count),
            pedestrian_count=(pedestrian_count),
            rider_count=(rider_count),
            traffic_light_count=(traffic_light_count),
            traffic_sign_count=(traffic_sign_count),
            total_objects=(total_objects),
            lane_detected=(lane_detected),
            drivable_area_detected=(drivable_area_detected),
            lane_pixels=(lane_pixels),
            drivable_pixels=(drivable_pixels),
            drivable_ratio=(drivable_ratio),
        )

    # ========================================================
    # Cleanup
    # ========================================================

    def close(
        self,
    ) -> None:
        """
        PanopticDrivingDetector currently does not necessarily
        require explicit cleanup, but keep a stable service
        interface for future backends.
        """

        close_fn = getattr(
            self.detector,
            "close",
            None,
        )

        if callable(close_fn):
            close_fn()

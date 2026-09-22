from __future__ import annotations

import numpy as np

from modules.cabin.snapshot import CabinPerceptionSnapshot
from modules.driving.perception.types import DrivingSceneResult
from modules.driving.perception_service import DrivingPerceptionService
from modules.observation import ObservationMetadata, ObservationSequencer


class _Detector:
    def detect(self, frame: np.ndarray) -> DrivingSceneResult:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        return DrivingSceneResult(
            objects=[],
            drivable_mask=mask,
            lane_mask=mask,
            inference_ms=0.1,
            preprocess_ms=0.1,
            postprocess_ms=0.1,
            total_ms=0.3,
        )


def test_road_service_attaches_metadata_to_each_successful_frame() -> None:
    service = DrivingPerceptionService.__new__(DrivingPerceptionService)
    service.work_width = 8
    service.work_height = 8
    service.detector = _Detector()
    service._observations = ObservationSequencer("driving_perception")
    service._started_at = 0.0
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    first = service.process_frame(frame)
    second = service.process_frame(frame)

    assert first.metadata.sequence == 0
    assert second.metadata.sequence == 1
    assert first.metadata.source == "driving_perception"
    assert first.metadata.confidence is None
    assert first.metadata.valid is True
    assert first.metadata.timestamp_ms >= 0
    assert first.metadata.processing_ms >= 0


def test_road_service_preserves_supplied_frame_timeline() -> None:
    service = DrivingPerceptionService.__new__(DrivingPerceptionService)
    service.work_width = 8
    service.work_height = 8
    service.detector = _Detector()
    service._observations = ObservationSequencer("driving_perception")
    service._started_at = 0.0
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    snapshot = service.process_frame(frame, timestamp_ms=1234)

    assert snapshot.metadata.timestamp_ms == 1234


def test_cabin_snapshot_retains_source_timestamp_and_metadata() -> None:
    metadata = ObservationMetadata(
        timestamp_ms=250,
        sequence=3,
        source="cabin_perception",
        confidence=None,
        valid=True,
        processing_ms=4.0,
    )

    snapshot = CabinPerceptionSnapshot(
        metadata=metadata,
        face_visible=False,
        presence="UNKNOWN",
        driver_state="UNKNOWN",
        risk="UNKNOWN",
        perclos=None,
        perclos_ready=False,
        eye_closed=None,
        eye_closure_seconds=0.0,
        recent_yawns=0,
        blink_count=0,
        current_yawn=False,
    )

    assert snapshot.timestamp_ms == 250
    assert snapshot.metadata.valid is True
    assert snapshot.eye_closed is None

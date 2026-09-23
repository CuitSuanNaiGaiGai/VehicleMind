from __future__ import annotations

import time

from collections import deque
from types import SimpleNamespace

import numpy as np

from apps.vehicle_ai_demo.video_pipelines import (
    build_cabin_pipeline,
    build_road_pipeline,
)
from apps.vehicle_ai_demo.video_source import VideoFrame
from modules.observation import ObservationSequencer
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime


class _Source:
    fps = 30.0

    def __init__(self) -> None:
        self.frames = deque(
            VideoFrame(
                np.zeros((2, 2, 3), dtype=np.uint8), index * 33, time.monotonic()
            )
            for index in range(3)
        )
        self.closed = False

    def read(self) -> VideoFrame | None:
        return self.frames.popleft() if self.frames else None

    def close(self) -> None:
        self.closed = True


class _CabinService:
    def __init__(self) -> None:
        self.observations = ObservationSequencer("cabin_perception")
        self.closed = False

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> SimpleNamespace:
        return SimpleNamespace(
            metadata=self.observations.next(timestamp_ms=timestamp_ms, processing_ms=1),
            to_context_kwargs=lambda: {
                "presence": "PRESENT",
                "driver_state": "NORMAL",
                "risk": "LOW",
            },
            presence="PRESENT",
            driver_state="NORMAL",
            face_visible=True,
        )

    def close(self) -> None:
        self.closed = True


class _RoadService:
    def __init__(self) -> None:
        self.observations = ObservationSequencer("driving_perception")
        self.closed = False

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> SimpleNamespace:
        return SimpleNamespace(
            metadata=self.observations.next(timestamp_ms=timestamp_ms, processing_ms=1),
            to_context_kwargs=lambda: {
                "vehicle_count": 2,
                "lane_detected": True,
                "drivable_area_detected": True,
            },
            vehicle_count=2,
            pedestrian_count=0,
            lane_detected=True,
            drivable_area_detected=True,
        )

    def close(self) -> None:
        self.closed = True


def _runtime() -> VehicleMindRuntime:
    return VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),))
    )


def test_cabin_video_pipeline_updates_shared_context() -> None:
    runtime = _runtime()
    source = _Source()
    service = _CabinService()
    cards: list[object] = []
    pipeline = build_cabin_pipeline(
        runtime,
        source=source,
        service=service,
        display=cards.append,
        inference_hz=100,
    )

    pipeline.start()
    pipeline.join(timeout=5)

    assert runtime.context_manager.get_context().driver.presence == "PRESENT"
    assert source.closed and service.closed
    assert cards
    assert pipeline.health()["event_publish_p95_ms"] is not None


def test_road_video_pipeline_updates_shared_context() -> None:
    runtime = _runtime()
    source = _Source()
    service = _RoadService()
    cards: list[object] = []
    pipeline = build_road_pipeline(
        runtime,
        source=source,
        service=service,
        display=cards.append,
        inference_hz=100,
    )

    pipeline.start()
    pipeline.join(timeout=5)

    assert runtime.context_manager.get_context().road.lane_detected is True
    assert source.closed and service.closed
    assert cards
    assert pipeline.health()["event_publish_p95_ms"] is not None

from __future__ import annotations

import time

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.vehicle_ai_demo.video_source import VideoFileSource, VideoFrame
from modules.config import PerceptionConfig
from modules.vehicle_ai.pipeline.runner import FrameSource, VideoPipeline
from modules.vehicle_ai.runtime import VehicleMindRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CABIN_VIDEO = PROJECT_ROOT / "assets/demo/MicroSleep_test.mp4"
DEFAULT_ROAD_VIDEO = PROJECT_ROOT / "assets/driving/road_test.mp4"
DEFAULT_CABIN_MODEL = PROJECT_ROOT / "models/mediapipe/face_landmarker.task"
DEFAULT_ROAD_MODEL = PROJECT_ROOT / "models/driving/YOLOPv2_512.onnx"


@dataclass(frozen=True)
class InferredFrame:
    snapshot: Any
    captured_at: float
    inferred_at: float


@dataclass(frozen=True)
class PerceptionCard:
    source: str
    summary: str
    event_count: int
    capture_to_context_ms: float
    perception_to_event_ms: float | None


def _display_every_two_seconds() -> Callable[[PerceptionCard], None]:
    last_print = 0.0

    def show(card: PerceptionCard) -> None:
        nonlocal last_print
        now = time.monotonic()
        if now - last_print >= 2.0:
            print(
                f"[{card.source}] {card.summary} events={card.event_count} "
                f"capture_to_context={card.capture_to_context_ms:.1f}ms"
            )
            last_print = now

    return show


def _source(
    supplied: FrameSource[VideoFrame] | None, path: Path
) -> FrameSource[VideoFrame]:
    return supplied if supplied is not None else VideoFileSource(path)


def _cabin_service(supplied: Any, model_path: Path) -> Any:
    if supplied is not None:
        return supplied
    if not model_path.is_file():
        raise FileNotFoundError(f"cabin model unavailable: {model_path}")
    from modules.cabin.perception_service import CabinPerceptionService

    return CabinPerceptionService(model_path=model_path)


def _road_service(supplied: Any, model_path: Path) -> Any:
    if supplied is not None:
        return supplied
    if not model_path.is_file():
        raise FileNotFoundError(f"road model unavailable: {model_path}")
    from modules.driving.perception_service import DrivingPerceptionService

    settings = PerceptionConfig.load_default().driving
    return DrivingPerceptionService(
        model_path=model_path,
        work_width=settings.work_width,
        work_height=settings.work_height,
        score_threshold=settings.score_threshold,
        nms_threshold=settings.nms_threshold,
        prefer_coreml=settings.prefer_coreml,
        warmup_runs=settings.warmup_runs,
    )


def build_cabin_pipeline(
    runtime: VehicleMindRuntime,
    *,
    video_path: Path = DEFAULT_CABIN_VIDEO,
    model_path: Path = DEFAULT_CABIN_MODEL,
    source: FrameSource[VideoFrame] | None = None,
    service: Any = None,
    display: Callable[[PerceptionCard], None] | None = None,
    inference_hz: float = 12.0,
    frame_capacity: int = 2,
) -> VideoPipeline[VideoFrame, InferredFrame, PerceptionCard]:
    video = _source(source, video_path)
    try:
        perception = _cabin_service(service, model_path)
    except Exception:
        video.close()
        raise

    def infer(frame: VideoFrame) -> InferredFrame:
        snapshot = perception.process_frame(
            frame.pixels, timestamp_ms=frame.timestamp_ms
        )
        return InferredFrame(snapshot, frame.captured_at, time.monotonic())

    def update(packet: InferredFrame) -> PerceptionCard:
        snapshot = packet.snapshot
        events = runtime.update_cabin(
            metadata=snapshot.metadata, **snapshot.to_context_kwargs()
        )
        now = time.monotonic()
        return PerceptionCard(
            source="Cabin",
            summary=(
                f"presence={snapshot.presence} state={snapshot.driver_state} "
                f"face={snapshot.face_visible}"
            ),
            event_count=len(events),
            capture_to_context_ms=(now - packet.captured_at) * 1000,
            perception_to_event_ms=(now - packet.inferred_at) * 1000
            if events
            else None,
        )

    return VideoPipeline(
        name="cabin",
        source=video,
        infer=infer,
        update=update,
        display=display or _display_every_two_seconds(),
        close_infer=perception.close,
        event_latency_ms=lambda card: card.perception_to_event_ms,
        frame_capacity=frame_capacity,
        capture_rate_hz=float(getattr(video, "fps", 30.0)),
        inference_rate_hz=inference_hz,
    )


def build_road_pipeline(
    runtime: VehicleMindRuntime,
    *,
    video_path: Path = DEFAULT_ROAD_VIDEO,
    model_path: Path = DEFAULT_ROAD_MODEL,
    source: FrameSource[VideoFrame] | None = None,
    service: Any = None,
    display: Callable[[PerceptionCard], None] | None = None,
    inference_hz: float = 8.0,
    frame_capacity: int = 2,
) -> VideoPipeline[VideoFrame, InferredFrame, PerceptionCard]:
    video = _source(source, video_path)
    try:
        perception = _road_service(service, model_path)
    except Exception:
        video.close()
        raise

    def infer(frame: VideoFrame) -> InferredFrame:
        snapshot = perception.process_frame(
            frame.pixels, timestamp_ms=frame.timestamp_ms
        )
        return InferredFrame(snapshot, frame.captured_at, time.monotonic())

    def update(packet: InferredFrame) -> PerceptionCard:
        snapshot = packet.snapshot
        events = runtime.update_driving(
            metadata=snapshot.metadata, **snapshot.to_context_kwargs()
        )
        now = time.monotonic()
        return PerceptionCard(
            source="Road",
            summary=(
                f"vehicles={snapshot.vehicle_count} "
                f"pedestrians={snapshot.pedestrian_count} "
                f"lane={snapshot.lane_detected} "
                f"drivable={snapshot.drivable_area_detected}"
            ),
            event_count=len(events),
            capture_to_context_ms=(now - packet.captured_at) * 1000,
            perception_to_event_ms=(now - packet.inferred_at) * 1000
            if events
            else None,
        )

    return VideoPipeline(
        name="road",
        source=video,
        infer=infer,
        update=update,
        display=display or _display_every_two_seconds(),
        close_infer=perception.close,
        event_latency_ms=lambda card: card.perception_to_event_ms,
        frame_capacity=frame_capacity,
        capture_rate_hz=float(getattr(video, "fps", 30.0)),
        inference_rate_hz=inference_hz,
    )

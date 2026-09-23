from __future__ import annotations

import time

from collections import deque
from threading import Thread

import pytest

from modules.vehicle_ai.pipeline.runner import VideoPipeline


class _FiniteSource:
    def __init__(self, count: int) -> None:
        self.frames = deque(range(count))
        self.closed = False

    def read(self) -> int | None:
        return self.frames.popleft() if self.frames else None

    def close(self) -> None:
        self.closed = True


def test_pipeline_bounds_frames_and_preserves_inferred_snapshots() -> None:
    source = _FiniteSource(60)
    displayed: list[int] = []

    def infer(frame: int) -> int:
        time.sleep(0.005)
        return frame * 2

    pipeline = VideoPipeline(
        name="road",
        source=source,
        infer=infer,
        update=lambda snapshot: snapshot,
        display=displayed.append,
        frame_capacity=2,
        snapshot_capacity=2,
        display_capacity=2,
        capture_rate_hz=1000,
        inference_rate_hz=1000,
    )

    pipeline.start()
    pipeline.join(timeout=5)
    health = pipeline.health()

    assert source.closed is True
    assert health["queues"]["frames"]["dropped"] > 0
    assert health["queues"]["frames"]["depth"] <= 2
    assert (
        health["stages"]["inference"]["processed"]
        == (health["stages"]["context"]["processed"])
    )
    assert health["stages"]["display"]["processed"] == len(displayed)
    assert health["last_error"] is None


def test_pipeline_supports_slower_inference_than_capture() -> None:
    source = _FiniteSource(30)
    pipeline = VideoPipeline(
        name="cabin",
        source=source,
        infer=lambda frame: frame,
        update=lambda snapshot: snapshot,
        display=lambda result: None,
        frame_capacity=2,
        capture_rate_hz=500,
        inference_rate_hz=20,
    )

    pipeline.start()
    pipeline.join(timeout=5)
    health = pipeline.health()

    assert health["stages"]["capture"]["processed"] == 30
    assert health["stages"]["inference"]["processed"] < 30
    assert health["queues"]["frames"]["dropped"] > 0


def test_pipeline_reports_event_publication_latency() -> None:
    pipeline = VideoPipeline(
        name="latency",
        source=_FiniteSource(2),
        infer=lambda frame: frame,
        update=lambda snapshot: 2.5,
        display=lambda result: None,
        event_latency_ms=lambda result: result,
        capture_rate_hz=100,
        inference_rate_hz=100,
    )

    pipeline.start()
    pipeline.join(timeout=5)

    assert pipeline.health()["event_publish_p95_ms"] == 2.5


def test_pipeline_records_inference_error_and_stops_all_stages() -> None:
    source = _FiniteSource(10)
    closed: list[bool] = []

    def fail(_: int) -> int:
        raise RuntimeError("inference failed")

    pipeline = VideoPipeline(
        name="fault",
        source=source,
        infer=fail,
        update=lambda snapshot: snapshot,
        display=lambda result: None,
        capture_rate_hz=100,
        inference_rate_hz=100,
        close_infer=lambda: closed.append(True),
    )

    pipeline.start()
    pipeline.join(timeout=5)
    health = pipeline.health()

    assert source.closed is True
    assert closed == [True]
    assert "inference failed" in health["last_error"]
    assert all(not thread.is_alive() for thread in pipeline.threads)


def test_pipeline_cleans_up_after_partial_thread_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _FiniteSource(10)
    closed: list[bool] = []
    original_start = Thread.start

    def fail_inference_start(thread: Thread) -> None:
        if thread.name == "partial-inference":
            raise RuntimeError("thread unavailable")
        original_start(thread)

    monkeypatch.setattr(Thread, "start", fail_inference_start)
    pipeline = VideoPipeline(
        name="partial",
        source=source,
        infer=lambda frame: frame,
        update=lambda snapshot: snapshot,
        display=lambda result: None,
        close_infer=lambda: closed.append(True),
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        pipeline.start()

    assert source.closed
    assert closed == [True]
    assert all(not thread.is_alive() for thread in pipeline.threads)

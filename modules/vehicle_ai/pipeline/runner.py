from __future__ import annotations

import time
import math

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Generic, Protocol, TypeVar

from modules.vehicle_ai.pipeline.channel import (
    BoundedChannel,
    ChannelClosed,
    OverflowPolicy,
)


FrameT = TypeVar("FrameT")
SourceT = TypeVar("SourceT", covariant=True)
SnapshotT = TypeVar("SnapshotT")
ResultT = TypeVar("ResultT")


class FrameSource(Protocol[SourceT]):
    def read(self) -> SourceT | None: ...
    def close(self) -> None: ...


@dataclass
class StageMetrics:
    processed: int = 0
    heartbeat_at: float | None = None
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=512))


class VideoPipeline(Generic[FrameT, SnapshotT, ResultT]):
    """Four independently scheduled stages with bounded inter-stage channels."""

    def __init__(
        self,
        *,
        name: str,
        source: FrameSource[FrameT],
        infer: Callable[[FrameT], SnapshotT],
        update: Callable[[SnapshotT], ResultT],
        display: Callable[[ResultT], None],
        event_latency_ms: Callable[[ResultT], float | None] | None = None,
        close_infer: Callable[[], None] | None = None,
        frame_capacity: int = 2,
        snapshot_capacity: int = 4,
        display_capacity: int = 2,
        capture_rate_hz: float = 30.0,
        inference_rate_hz: float = 10.0,
    ) -> None:
        if not name:
            raise ValueError("pipeline name must be nonempty")
        if any(
            not math.isfinite(rate) or rate <= 0
            for rate in (capture_rate_hz, inference_rate_hz)
        ):
            raise ValueError("stage rates must be positive")
        self.name = name
        self.source = source
        self.infer = infer
        self.update = update
        self.display = display
        self.event_latency_ms = event_latency_ms
        self.close_infer = close_infer
        self.capture_period = 1.0 / capture_rate_hz
        self.inference_period = 1.0 / inference_rate_hz
        self.frames: BoundedChannel[FrameT] = BoundedChannel(
            frame_capacity, OverflowPolicy.DROP_OLDEST
        )
        self.snapshots: BoundedChannel[SnapshotT] = BoundedChannel(
            snapshot_capacity, OverflowPolicy.BLOCK
        )
        self.displays: BoundedChannel[ResultT] = BoundedChannel(
            display_capacity, OverflowPolicy.DROP_OLDEST
        )
        self._stop = Event()
        self._lock = Lock()
        self._error: str | None = None
        self._event_latencies_ms: deque[float] = deque(maxlen=512)
        self._metrics: dict[str, StageMetrics] = {
            stage: StageMetrics()
            for stage in ("capture", "inference", "context", "display")
        }
        self.threads: tuple[Thread, ...] = ()

    def start(self) -> None:
        if self.threads:
            raise RuntimeError("pipeline already started")
        self.threads = tuple(
            Thread(target=target, name=f"{self.name}-{stage}", daemon=True)
            for stage, target in (
                ("capture", self._capture),
                ("inference", self._inference),
                ("context", self._context),
                ("display", self._display),
            )
        )
        started: list[Thread] = []
        try:
            for thread in self.threads:
                thread.start()
                started.append(thread)
        except Exception:
            self.stop()
            if not any(thread.name.endswith("-capture") for thread in started):
                self.source.close()
            if not any(thread.name.endswith("-inference") for thread in started):
                if self.close_infer is not None:
                    self.close_infer()
            self.threads = tuple(started)
            for thread in started:
                thread.join(timeout=5)
            raise

    def stop(self) -> None:
        self._stop.set()
        self.frames.close()
        self.snapshots.close()
        self.displays.close()

    def join(self, timeout: float | None = None) -> None:
        if not self.threads:
            raise RuntimeError("pipeline has not started")
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in self.threads:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            thread.join(remaining)
        if any(thread.is_alive() for thread in self.threads):
            raise TimeoutError(f"pipeline {self.name} did not stop within timeout")

    def _record(self, stage: str, started: float) -> None:
        with self._lock:
            metrics = self._metrics[stage]
            metrics.processed += 1
            metrics.heartbeat_at = time.monotonic()
            metrics.latencies_ms.append((time.monotonic() - started) * 1000)

    def _fail(self, stage: str, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = f"{stage}: {type(error).__name__}: {error}"
        self.stop()

    def _capture(self) -> None:
        next_due = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                item = self.source.read()
                if item is None:
                    break
                self.frames.put(item)
                self._record("capture", started)
                next_due += self.capture_period
                self._stop.wait(max(0.0, next_due - time.monotonic()))
        except ChannelClosed:
            pass
        except Exception as error:
            self._fail("capture", error)
        finally:
            try:
                self.source.close()
            except Exception as error:
                self._fail("capture_close", error)
            finally:
                self.frames.close()

    def _inference(self) -> None:
        next_due = time.monotonic()
        try:
            while not self._stop.is_set():
                self._stop.wait(max(0.0, next_due - time.monotonic()))
                if self._stop.is_set():
                    break
                frame = self.frames.get()
                started = time.monotonic()
                snapshot = self.infer(frame)
                self.snapshots.put(snapshot)
                self._record("inference", started)
                next_due = time.monotonic() + self.inference_period
        except ChannelClosed:
            pass
        except Exception as error:
            self._fail("inference", error)
        finally:
            try:
                if self.close_infer is not None:
                    self.close_infer()
            except Exception as error:
                self._fail("inference_close", error)
            finally:
                self.snapshots.close()

    def _context(self) -> None:
        try:
            while not self._stop.is_set():
                snapshot = self.snapshots.get()
                started = time.monotonic()
                display_item = self.update(snapshot)
                if self.event_latency_ms is not None:
                    latency = self.event_latency_ms(display_item)
                    if latency is not None:
                        with self._lock:
                            self._event_latencies_ms.append(latency)
                self.displays.put(display_item)
                self._record("context", started)
        except ChannelClosed:
            pass
        except Exception as error:
            self._fail("context", error)
        finally:
            self.displays.close()

    def _display(self) -> None:
        try:
            while not self._stop.is_set():
                item = self.displays.get()
                started = time.monotonic()
                self.display(item)
                self._record("display", started)
        except ChannelClosed:
            pass
        except Exception as error:
            self._fail("display", error)

    def health(self) -> dict[str, object]:
        with self._lock:
            stages = {}
            for stage, metrics in self._metrics.items():
                latency = sorted(metrics.latencies_ms)
                p95 = latency[int(0.95 * (len(latency) - 1))] if latency else None
                stages[stage] = {
                    "processed": metrics.processed,
                    "heartbeat_at": metrics.heartbeat_at,
                    "p95_latency_ms": p95,
                }
            error = self._error
            event_latencies = sorted(self._event_latencies_ms)
            event_p95 = (
                event_latencies[int(0.95 * (len(event_latencies) - 1))]
                if event_latencies
                else None
            )
        return {
            "name": self.name,
            "last_error": error,
            "event_publish_p95_ms": event_p95,
            "stages": stages,
            "queues": {
                "frames": {"depth": self.frames.depth, "dropped": self.frames.dropped},
                "snapshots": {
                    "depth": self.snapshots.depth,
                    "dropped": self.snapshots.dropped,
                },
                "display": {
                    "depth": self.displays.depth,
                    "dropped": self.displays.dropped,
                },
            },
        }

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock
from threading import Event, Thread
from typing import Any, Protocol
import time


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    thread_id: str
    task_id: str
    event_type: str
    timestamp_ms: int
    graph_node: str | None
    latency_ms: float | None
    attributes: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "thread_id": self.thread_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "timestamp_ms": self.timestamp_ms,
            "graph_node": self.graph_node,
            "latency_ms": self.latency_ms,
            "attributes": deepcopy(dict(self.attributes)),
        }


class TraceBackend(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class AgentTraceRecorder:
    """Collect structured events locally and isolate failures in optional exporters."""

    def __init__(
        self,
        thread_id: str,
        *,
        backend: TraceBackend | None = None,
        task_id_provider: Callable[[], str] | None = None,
        max_events: int = 2048,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.thread_id = thread_id
        self.backend = backend
        self.task_id_provider = task_id_provider or (lambda: "unassigned")
        self.max_events = max_events
        self.clock = clock
        self._events: list[TraceEvent] = []
        self._backend_errors: list[dict[str, str]] = []
        self._sequence = 0
        self._lock = Lock()
        self._exporter = _TraceExportDispatcher() if backend is not None else None

    def emit(
        self,
        event_type: str,
        *,
        graph_node: str | None = None,
        latency_ms: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            task_id = str(self.task_id_provider() or "unassigned")
            with self._lock:
                event = TraceEvent(
                    sequence=self._sequence,
                    thread_id=self.thread_id,
                    task_id=task_id,
                    event_type=event_type,
                    timestamp_ms=int(self.clock() * 1000),
                    graph_node=graph_node,
                    latency_ms=latency_ms,
                    attributes=deepcopy(dict(attributes or {})),
                )
                self._sequence += 1
                self._events.append(event)
                del self._events[: -self.max_events]
                if self._exporter is not None:
                    error_type = self._exporter.submit(self, event)
                    if error_type is not None:
                        self._record_backend_error_locked(event_type, error_type)
        except Exception as error:
            # Tracing is an observation side channel and must never fail the agent.
            with self._lock:
                self._record_backend_error_locked(event_type, type(error).__name__)

    def _record_backend_error_locked(self, event_type: str, error_type: str) -> None:
        self._backend_errors.append(
            {"event_type": event_type, "error_type": error_type}
        )
        del self._backend_errors[: -self.max_events]

    def _record_backend_error(self, event_type: str, error_type: str) -> None:
        with self._lock:
            self._record_backend_error_locked(event_type, error_type)

    def flush_backend(self, timeout_seconds: float = 1.0) -> bool:
        """Wait for events queued before this call, bounded by the supplied timeout."""
        if self._exporter is None:
            return True
        return self._exporter.flush(timeout_seconds)

    def records(self) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(event.to_dict() for event in self.records())

    @property
    def backend_errors(self) -> tuple[dict[str, str], ...]:
        with self._lock:
            return tuple(deepcopy(self._backend_errors))


@dataclass(frozen=True)
class _ExportItem:
    recorder: AgentTraceRecorder
    backend: TraceBackend
    event: TraceEvent


@dataclass(frozen=True)
class _FlushMarker:
    completed: Event


class _TraceExportDispatcher:
    """Per-recorder bounded daemon exporter isolates backend stalls."""

    def __init__(self, max_queue_size: int = 256) -> None:
        self._queue: Queue[_ExportItem | _FlushMarker] = Queue(maxsize=max_queue_size)
        self._lock = Lock()
        self._worker: Thread | None = None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = Thread(
                    target=self._run,
                    name="vehiclemind-trace-export",
                    daemon=True,
                )
                self._worker.start()

    def submit(self, recorder: AgentTraceRecorder, event: TraceEvent) -> str | None:
        try:
            self._ensure_worker()
            self._queue.put_nowait(
                _ExportItem(recorder, recorder.backend, deepcopy(event))
            )
        except Full:
            return "TraceQueueFull"
        except Exception as error:
            return type(error).__name__
        return None

    def flush(self, timeout_seconds: float) -> bool:
        marker = _FlushMarker(Event())
        deadline = time.monotonic() + max(timeout_seconds, 0)
        try:
            self._ensure_worker()
            self._queue.put(
                marker,
                timeout=max(deadline - time.monotonic(), 0),
            )
        except Exception:
            return False
        return marker.completed.wait(max(deadline - time.monotonic(), 0))

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=1)
            except Empty:
                continue
            try:
                if isinstance(item, _FlushMarker):
                    item.completed.set()
                    continue
                try:
                    item.backend.emit(item.event)
                except Exception as error:
                    item.recorder._record_backend_error(
                        item.event.event_type, type(error).__name__
                    )
            finally:
                self._queue.task_done()

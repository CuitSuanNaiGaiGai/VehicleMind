"""Bounded decisions for no-tools recommendations from trusted driver events."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import RLock
from dataclasses import dataclass

from modules.vehicle_ai.agent.policy import RecommendationPolicy
from modules.vehicle_ai.agent.vehicle_agent import VehicleAgent
from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.context.enums import RiskLevel
from modules.vehicle_ai.context.quality import QualityStatus
from modules.vehicle_ai.events import EventType, VehicleEvent


@dataclass(frozen=True)
class RecommendationTrigger:
    event_id: str
    event_time: float
    context_quality: QualityStatus | None
    reason: str
    model_result: str | None = None
    error: str | None = None


class RecommendationCoordinator:
    """Deduplicate and cool down trusted high-risk event recommendations."""

    def __init__(
        self,
        agent: VehicleAgent,
        context_manager: ContextManager,
        policy: RecommendationPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_trace_events: int = 200,
        max_seen_event_ids: int = 2048,
    ) -> None:
        if max_trace_events < 1 or max_seen_event_ids < 1:
            raise ValueError("recommendation limits must be positive")
        self.agent = agent
        self.context_manager = context_manager
        self.policy = policy
        self.clock = clock
        self.max_trace_events = max_trace_events
        self.max_seen_event_ids = max_seen_event_ids
        self.trace: list[RecommendationTrigger] = []
        self._seen_event_ids: dict[str, None] = {}
        self._last_triggered: dict[tuple[EventType, str], float] = {}
        self._lock = RLock()

    def _remember_event(self, event_id: str) -> None:
        with self._lock:
            self._seen_event_ids[event_id] = None
            if len(self._seen_event_ids) > self.max_seen_event_ids:
                oldest = next(iter(self._seen_event_ids))
                del self._seen_event_ids[oldest]

    def _record(
        self,
        event: VehicleEvent,
        quality: QualityStatus | None,
        reason: str,
        result: str | None = None,
        error: str | None = None,
    ) -> RecommendationTrigger:
        trigger = RecommendationTrigger(
            event_id=event.event_id,
            event_time=event.timestamp,
            context_quality=quality,
            reason=reason,
            model_result=result,
            error=error,
        )
        with self._lock:
            self.trace.append(trigger)
            del self.trace[: -self.max_trace_events]
        return trigger

    def record_failure(
        self, event: VehicleEvent, error: Exception
    ) -> RecommendationTrigger:
        """Preserve a failure when a subscriber catches an unexpected exception."""
        return self._record(event, None, "RECOMMENDATION_ERROR", error=str(error))

    def record_suppressed(
        self, event: VehicleEvent, reason: str
    ) -> RecommendationTrigger:
        """Record a runtime-level suppression without invoking the model."""
        self._remember_event(event.event_id)
        return self._record(event, None, reason)

    def on_event(self, event: VehicleEvent) -> RecommendationTrigger:
        if not self.policy.enabled or event.type is not EventType.HIGH_RISK_DETECTED:
            return self._record(event, None, "DISABLED")
        with self._lock:
            if event.event_id in self._seen_event_ids:
                return self._record(event, None, "DUPLICATE_SUPPRESSED")
            self._remember_event(event.event_id)

        snapshot, qualities = self.context_manager.snapshot_with_quality(
            (("driver", "risk"),)
        )
        quality = qualities[("driver", "risk")]
        if (
            snapshot.driver.risk is not RiskLevel.HIGH
            or quality is not QualityStatus.KNOWN
            or event.data.get("risk") != RiskLevel.HIGH
        ):
            return self._record(event, quality, "INVALID_CONTEXT")

        key = (event.type, str(event.data["risk"]))
        now = self.clock()
        with self._lock:
            previous = self._last_triggered.get(key)
            if (
                previous is not None
                and (now - previous) * 1000 < self.policy.cooldown_ms
            ):
                return self._record(event, quality, "COOLDOWN_SUPPRESSED")
            # Reserve the cooldown before the request, so failures cannot create a
            # rapid retry loop and concurrent events cannot both pass the gate.
            self._last_triggered[key] = now
        try:
            result = self.agent.recommend_from_event(event)
        except Exception as error:
            return self._record(
                event, quality, "RECOMMENDATION_ERROR", error=str(error)
            )
        return self._record(event, quality, "TRIGGERED", result=result)

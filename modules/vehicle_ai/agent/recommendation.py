"""Bounded decisions for no-tools recommendations from trusted driver events."""

from __future__ import annotations

import time
from collections.abc import Callable
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
    ) -> None:
        if max_trace_events < 1:
            raise ValueError("max_trace_events must be positive")
        self.agent = agent
        self.context_manager = context_manager
        self.policy = policy
        self.clock = clock
        self.max_trace_events = max_trace_events
        self.trace: list[RecommendationTrigger] = []
        self._seen_event_ids: set[str] = set()
        self._last_triggered: dict[tuple[EventType, str], float] = {}

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
        self.trace.append(trigger)
        del self.trace[: -self.max_trace_events]
        return trigger

    def record_failure(
        self, event: VehicleEvent, error: Exception
    ) -> RecommendationTrigger:
        """Preserve a failure when a subscriber catches an unexpected exception."""
        return self._record(event, None, "RECOMMENDATION_ERROR", error=str(error))

    def on_event(self, event: VehicleEvent) -> RecommendationTrigger:
        if not self.policy.enabled or event.type is not EventType.HIGH_RISK_DETECTED:
            return self._record(event, None, "DISABLED")
        if event.event_id in self._seen_event_ids:
            return self._record(event, None, "DUPLICATE_SUPPRESSED")
        self._seen_event_ids.add(event.event_id)

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
        previous = self._last_triggered.get(key)
        if previous is not None and (now - previous) * 1000 < self.policy.cooldown_ms:
            return self._record(event, quality, "COOLDOWN_SUPPRESSED")
        try:
            result = self.agent.recommend_from_event(event)
        except Exception as error:
            return self._record(
                event, quality, "RECOMMENDATION_ERROR", error=str(error)
            )
        self._last_triggered[key] = now
        return self._record(event, quality, "TRIGGERED", result=result)

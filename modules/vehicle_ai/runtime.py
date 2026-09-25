from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from threading import Lock

from modules.vehicle_ai.agent import (
    VehicleAgent,
)
from modules.vehicle_ai.agent.policy import AgentPolicy, RecommendationPolicy
from modules.vehicle_ai.agent.recommendation import RecommendationCoordinator

from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.events import (
    EventBus,
    EventDetector,
    EventType,
    VehicleEvent,
)

from modules.vehicle_ai.integration import (
    CabinContextAdapter,
    DrivingContextAdapter,
)

from modules.vehicle_ai.llm import (
    BaseLLMClient,
)

from modules.vehicle_ai.tools import (
    ToolRegistry,
    build_default_tool_registry,
)
from modules.observation import ObservationMetadata
from modules.config.events import EventTimingConfig


class VehicleMindRuntime:
    """
    Unified VehicleMind runtime.

    There must be exactly ONE shared ContextManager for:

        Cabin Perception
        Driving Perception
        Vehicle Tools
        Vehicle Agent

    This is the central integration point of VehicleMind.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        event_timing: EventTimingConfig | None = None,
        max_tool_rounds: int = 5,
        quality_clock: Callable[[], float] = time.monotonic,
        action_clock: Callable[[], float] = time.time,
        turn_timeout_seconds: float = 90.0,
        max_tool_calls: int = 10,
        max_task_trace_events: int = 200,
        enable_event_recommendations: bool = True,
    ):
        # ====================================================
        # Shared context
        # ====================================================

        self.context_manager = ContextManager(quality_clock=quality_clock)

        # ====================================================
        # Perception adapters
        # ====================================================

        self.cabin = CabinContextAdapter(self.context_manager)

        self.driving = DrivingContextAdapter(self.context_manager)

        # ====================================================
        # Semantic event infrastructure
        # ====================================================

        self.event_detector = EventDetector(timing=event_timing)

        self.event_bus = EventBus()

        # ====================================================
        # Vehicle tools
        # ====================================================

        self.tools: ToolRegistry = build_default_tool_registry(self.context_manager)

        # ====================================================
        # Vehicle Agent
        # ====================================================

        self.agent = VehicleAgent(
            llm=llm,
            context_manager=(self.context_manager),
            tool_registry=(self.tools),
            max_tool_rounds=max_tool_rounds,
            action_clock=action_clock,
            turn_timeout_seconds=turn_timeout_seconds,
            max_tool_calls=max_tool_calls,
            max_task_trace_events=max_task_trace_events,
        )
        recommendation = AgentPolicy.from_yaml().recommendation
        if not enable_event_recommendations:
            recommendation = RecommendationPolicy(
                enabled=False, cooldown_ms=recommendation.cooldown_ms
            )
        self.recommendation_coordinator = RecommendationCoordinator(
            self.agent,
            self.context_manager,
            recommendation,
        )
        self.event_bus.subscribe(
            EventType.HIGH_RISK_DETECTED, self._recommend_from_event
        )
        self._recommendation_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vehiclemind-recommendation"
        )
        self._recommendation_lock = Lock()
        self._recommendation_future: Future[None] | None = None

    def _recommend_from_event(self, event: VehicleEvent) -> None:
        if not self.recommendation_coordinator.policy.enabled:
            self.recommendation_coordinator.on_event(event)
            return
        with self._recommendation_lock:
            if (
                self._recommendation_future is not None
                and not self._recommendation_future.done()
            ):
                self.recommendation_coordinator.record_suppressed(
                    event, "IN_FLIGHT_SUPPRESSED"
                )
                return
            self._recommendation_future = self._recommendation_executor.submit(
                self._run_recommendation, event
            )

    def _run_recommendation(self, event: VehicleEvent) -> None:
        try:
            self.recommendation_coordinator.on_event(event)
        except Exception as error:
            self.recommendation_coordinator.record_failure(event, error)

    def wait_for_recommendations(self, timeout_seconds: float | None = None) -> None:
        """Wait for the currently scheduled recommendation, mainly for replay/tests."""
        with self._recommendation_lock:
            future = self._recommendation_future
        if future is not None:
            future.result(timeout=timeout_seconds)

    # ========================================================
    # Context-change processing
    # ========================================================

    def _process_changes(
        self,
        changes,
        *,
        observed_domain: str | None = None,
        at_ms: int | None = None,
    ):
        if not changes and observed_domain is None:
            return []

        context = self.context_manager.get_context()

        events = self.event_detector.detect(
            changes=changes,
            context=context,
        )
        if observed_domain is not None:
            observed_at = time.monotonic_ns() // 1_000_000 if at_ms is None else at_ms
            events.extend(
                self.event_detector.observe_hazards(
                    observed_domain, context, at_ms=observed_at
                )
            )

        self.event_bus.publish_many(events)

        return events

    # ========================================================
    # Cabin Perception
    # ========================================================

    def update_cabin(
        self,
        metadata: ObservationMetadata | None = None,
        at_ms: int | None = None,
        **kwargs,
    ):
        if metadata is not None and not metadata.valid:
            self.context_manager.mark_invalid_observation("driver", metadata)
            self.event_detector.invalidate_observation("driver")
            return []
        changes = self.cabin.update(observation=metadata, **kwargs)

        return self._process_changes(changes, observed_domain="driver", at_ms=at_ms)

    # ========================================================
    # Driving Perception
    # ========================================================

    def update_driving(
        self,
        metadata: ObservationMetadata | None = None,
        at_ms: int | None = None,
        **kwargs,
    ):
        if metadata is not None and not metadata.valid:
            self.context_manager.mark_invalid_observation("road", metadata)
            self.event_detector.invalidate_observation("road")
            return []
        changes = self.driving.update(observation=metadata, **kwargs)

        return self._process_changes(changes, observed_domain="road", at_ms=at_ms)

    def update_vehicle(
        self,
        metadata: ObservationMetadata | None = None,
        **kwargs,
    ):
        if metadata is not None and not metadata.valid:
            self.context_manager.mark_invalid_observation("vehicle", metadata)
            return []
        changes = self.context_manager.update_vehicle(observation=metadata, **kwargs)

        return self._process_changes(changes)

    # ========================================================
    # Agent
    # ========================================================

    def chat(
        self,
        text: str,
        debug: bool = True,
    ) -> str:

        return self.agent.chat(
            text,
            debug=debug,
        )

    # ========================================================
    # Context snapshot
    # ========================================================

    def context_summary(
        self,
    ) -> str:

        return self.context_manager.summary()

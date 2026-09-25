from __future__ import annotations

import time
import uuid
import sqlite3
from dataclasses import replace
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from modules.vehicle_ai.agent import (
    VehicleAgent,
)
from modules.vehicle_ai.agent.policy import AgentPolicy, RecommendationPolicy
from modules.vehicle_ai.agent.recommendation import RecommendationCoordinator
from modules.vehicle_ai.agent.session import record

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
from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog
from modules.vehicle_ai.knowledge.lightrag_client import LightRAGClient
from modules.vehicle_ai.knowledge.profile_router import ProfileRouter
from modules.vehicle_ai.knowledge.tool import KnowledgeClient, build_knowledge_tool
from modules.vehicle_ai.memory import TripEvent, TripEventStore, build_trip_memory_tool
from modules.vehicle_ai.memory.trace_persistence import trace_to_trip_event

from modules.vehicle_ai.tools import (
    NavigationConfig,
    ToolDefinition,
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
        knowledge_profile: str | None = None,
        knowledge_client: KnowledgeClient | None = None,
        trip_event_store: TripEventStore | None = None,
        trip_id: str | None = None,
        trip_event_retention_days: int = 30,
        navigation_config: NavigationConfig | None = None,
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
        self.trip_event_store = trip_event_store
        self.trip_memory_errors: list[str] = []
        self.trip_id = trip_id or uuid.uuid4().hex
        if trip_event_store is not None:
            if not self.trip_id.strip():
                raise ValueError("trip_id must not be empty")
            if trip_event_retention_days < 1:
                raise ValueError("trip_event_retention_days must be positive")
            trip_event_store.prune(
                before=time.time() - trip_event_retention_days * 86400
            )
            self.event_bus.subscribe_all(self._persist_vehicle_event)

        # ====================================================
        # Vehicle tools
        # ====================================================

        knowledge_tools: tuple[ToolDefinition, ...] = ()
        if knowledge_profile is not None:
            router = ProfileRouter()
            effective_profile, _ = router.resolve(knowledge_profile)
            if knowledge_client is None:
                catalog_path = (
                    Path(__file__).resolve().parents[2]
                    / "config"
                    / "knowledge"
                    / "source_catalog.yaml"
                )
                sources = KnowledgeCatalog().load(catalog_path)
                knowledge_client = LightRAGClient(router, sources)
            knowledge_tools = (
                build_knowledge_tool(
                    knowledge_client, self.context_manager, effective_profile
                ),
            )
        memory_tools = (
            (build_trip_memory_tool(trip_event_store, self.trip_id),)
            if trip_event_store is not None
            else ()
        )
        self.tools: ToolRegistry = build_default_tool_registry(
            self.context_manager,
            extra_tools=knowledge_tools + memory_tools,
            navigation_config=navigation_config,
        )

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
            trip_id=self.trip_id if trip_event_store is not None else None,
            historical_event_sink=(
                self._persist_agent_trace if trip_event_store is not None else None
            ),
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
            trigger = self.recommendation_coordinator.on_event(event)
            self._persist_recommendation(trigger, event)
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
            enriched = self._attach_prior_trip_interactions(event)
            trigger = self.recommendation_coordinator.on_event(enriched)
            self._record_fallback_recommendation(trigger, event)
            self._persist_recommendation(trigger, event)
        except Exception as error:
            trigger = self.recommendation_coordinator.record_failure(event, error)
            self._record_fallback_recommendation(trigger, event)
            self._persist_recommendation(trigger, event)

    def _record_fallback_recommendation(self, trigger, event: VehicleEvent) -> None:
        if not trigger.fallback_message:
            return
        record(
            self.agent,
            "event_recommendation",
            source="deterministic_fallback",
            quality="DETERMINISTIC",
            event_id=event.event_id,
            text=trigger.fallback_message,
            error=trigger.error,
        )

    def _attach_prior_trip_interactions(self, event: VehicleEvent) -> VehicleEvent:
        if self.trip_event_store is None:
            return event
        try:
            history = self.trip_event_store.query(trip_id=self.trip_id, limit=50)
        except (OSError, sqlite3.Error, ValueError) as error:
            self._record_trip_memory_error(error)
            return event
        interactions = []
        relevant_types = {
            "REMINDER",
            "USER_SELECTION",
            "ACTION_CANCELLED",
            "ACTION_CONFIRMED",
            "ACTION_OUTCOME",
        }
        for item in history:
            if item.event_type not in relevant_types:
                continue
            payload = item.payload
            if item.event_type == "REMINDER":
                summary = str(payload.get("message", ""))
            elif item.event_type == "USER_SELECTION":
                summary = str(payload.get("selected_target") or "用户更改了目的地")
            elif item.event_type in {"ACTION_CANCELLED", "ACTION_CONFIRMED"}:
                action = payload.get("action") or {}
                summary = str(
                    action.get("display_text") or action.get("tool_name") or "车机操作"
                )
            else:
                result = payload.get("result") or {}
                summary = str(
                    result.get("message") or payload.get("source") or "动作结果"
                )
            interactions.append(
                {
                    "event_type": item.event_type,
                    "occurred_at": item.occurred_at,
                    "summary": summary[:240],
                }
            )
        return replace(
            event,
            data={
                **event.data,
                "prior_trip_interactions": interactions[-3:],
            },
        )

    def _persist_vehicle_event(self, event: VehicleEvent) -> None:
        if self.trip_event_store is None:
            return
        if event.type not in {
            EventType.HIGH_RISK_DETECTED,
            EventType.DRIVER_RISK_CHANGED,
            EventType.LANE_LOST,
            EventType.DRIVABLE_AREA_LOST,
        }:
            return
        self._append_trip_event(
            TripEvent(
                event_id=event.event_id,
                trip_id=self.trip_id,
                event_type="RISK",
                occurred_at=event.timestamp,
                source=event.source,
                payload={
                    "semantic_type": event.type.value,
                    "priority": event.priority.value,
                    "message": event.message,
                    "data": event.data,
                },
            )
        )

    def _persist_agent_trace(self, trace: dict) -> None:
        if self.trip_event_store is None:
            return
        event = trace_to_trip_event(self.trip_id, trace)
        if event is not None:
            self._append_trip_event(event)

    def _persist_recommendation(self, trigger, event: VehicleEvent) -> None:
        if self.trip_event_store is None or not (
            trigger.reason == "TRIGGERED" or trigger.fallback_message
        ):
            return
        fallback = trigger.fallback_message is not None
        self._append_trip_event(
            TripEvent(
                event_id=f"reminder:{event.event_id}",
                trip_id=self.trip_id,
                event_type="REMINDER",
                occurred_at=event.timestamp,
                source=(
                    "deterministic_safety_fallback"
                    if fallback
                    else "event_recommendation_agent"
                ),
                payload={
                    "event_id": event.event_id,
                    "message": trigger.fallback_message or trigger.model_result,
                    "recommendation_source": (
                        "deterministic_fallback" if fallback else "model"
                    ),
                    "evidence": trigger.evidence or {},
                },
            )
        )

    def _append_trip_event(self, event: TripEvent) -> None:
        if self.trip_event_store is None:
            return
        try:
            self.trip_event_store.append(event)
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            self._record_trip_memory_error(error)

    def _record_trip_memory_error(self, error: Exception) -> None:
        self.trip_memory_errors.append(type(error).__name__)
        del self.trip_memory_errors[:-20]

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

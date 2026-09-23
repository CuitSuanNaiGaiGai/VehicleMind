from __future__ import annotations

import time

from modules.vehicle_ai.agent import (
    VehicleAgent,
)

from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.events import (
    EventBus,
    EventDetector,
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
    ):
        # ====================================================
        # Shared context
        # ====================================================

        self.context_manager = ContextManager()

        # ====================================================
        # Perception adapters
        # ====================================================

        self.cabin = CabinContextAdapter(self.context_manager)

        self.driving = DrivingContextAdapter(self.context_manager)

        # ====================================================
        # Semantic event infrastructure
        # ====================================================

        self.event_detector = EventDetector()

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
        )

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
        **kwargs,
    ):
        changes = self.context_manager.update_vehicle(**kwargs)

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

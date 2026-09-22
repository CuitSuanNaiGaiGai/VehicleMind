from __future__ import annotations

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
    ):
        if not changes:
            return []

        context = self.context_manager.get_context()

        events = self.event_detector.detect(
            changes=changes,
            context=context,
        )

        self.event_bus.publish_many(events)

        return events

    # ========================================================
    # Cabin Perception
    # ========================================================

    def update_cabin(
        self,
        **kwargs,
    ):
        changes = self.cabin.update(**kwargs)

        return self._process_changes(changes)

    # ========================================================
    # Driving Perception
    # ========================================================

    def update_driving(
        self,
        **kwargs,
    ):
        changes = self.driving.update(**kwargs)

        return self._process_changes(changes)

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

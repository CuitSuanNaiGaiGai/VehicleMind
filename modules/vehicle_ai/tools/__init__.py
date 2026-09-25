from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolExecutionRecord,
    ToolResult,
)

from modules.vehicle_ai.tools.registry import (
    ConfirmationIssuer,
    ToolRegistry,
)

from modules.vehicle_ai.tools.climate import (
    build_climate_tools,
)

from modules.vehicle_ai.tools.media import (
    build_media_tools,
)

from modules.vehicle_ai.tools.navigation import (
    build_navigation_tools,
)

from modules.vehicle_ai.tools.vehicle import (
    build_vehicle_tools,
)


# ============================================================
# Default Tool Registry
# ============================================================


def build_default_tool_registry(
    context_manager: ContextManager,
) -> ToolRegistry:

    from modules.vehicle_ai.agent.policy import AgentPolicy, PolicyContext

    def policy_context(user_intent: str) -> PolicyContext:
        snapshot, qualities = context_manager.snapshot_with_quality(
            (("driver", "risk"), ("vehicle", "speed_kmh"))
        )
        return PolicyContext(
            driver_risk=snapshot.driver.risk,
            driver_quality=qualities[("driver", "risk")],
            vehicle_moving=(
                qualities[("vehicle", "speed_kmh")] == "KNOWN"
                and snapshot.vehicle.speed_kmh > 0
            ),
            user_intent=user_intent,
        )

    registry = ToolRegistry(AgentPolicy.from_yaml(), policy_context)

    all_tools = (
        build_climate_tools(context_manager)
        + build_media_tools(context_manager)
        + build_navigation_tools(context_manager)
        + build_vehicle_tools(context_manager)
    )

    for tool in all_tools:
        tool.read_only = tool.name in {
            "get_vehicle_status",
            "get_climate_status",
            "get_media_status",
            "search_nearby_rest_area",
        }
        registry.register(tool)

    return registry


__all__ = [
    "ToolDefinition",
    "ToolExecutionRecord",
    "ToolResult",
    "ConfirmationIssuer",
    "ToolRegistry",
    "build_default_tool_registry",
]

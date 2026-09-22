from __future__ import annotations

from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolResult,
)


# ============================================================
# Vehicle Tools
# ============================================================


class VehicleTools:
    def __init__(
        self,
        context_manager: ContextManager,
    ):
        self.context_manager = context_manager

    # ========================================================
    # Status
    # ========================================================

    def get_vehicle_status(
        self,
    ) -> ToolResult:

        context = self.context_manager.get_context()

        vehicle = context.vehicle

        return ToolResult(
            success=True,
            message=("Vehicle status retrieved."),
            data={
                "speed_kmh": vehicle.speed_kmh,
                "gear": vehicle.gear,
                "driver_window_open": vehicle.driver_window_open,
                "passenger_window_open": vehicle.passenger_window_open,
                "volume": vehicle.volume,
                "navigation_state": vehicle.navigation_state,
            },
        )

    # ========================================================
    # Driver window
    # ========================================================

    def set_driver_window(
        self,
        open: bool,
    ) -> ToolResult:

        is_open = bool(open)

        self.context_manager.update_vehicle(driver_window_open=(is_open))

        return ToolResult(
            success=True,
            message=("Driver window " + ("opened." if is_open else "closed.")),
            data={
                "driver_window_open": is_open,
            },
        )


# ============================================================
# Registration
# ============================================================


def build_vehicle_tools(
    context_manager: ContextManager,
) -> list[ToolDefinition]:

    tools = VehicleTools(context_manager)

    return [
        ToolDefinition(
            name="get_vehicle_status",
            description=(
                "Get current vehicle status including speed, gear and window state."
            ),
            category="vehicle",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=(tools.get_vehicle_status),
        ),
        ToolDefinition(
            name="set_driver_window",
            description=("Open or close the driver-side window."),
            category="vehicle",
            parameters={
                "type": "object",
                "properties": {
                    "open": {
                        "type": "boolean",
                    },
                },
                "required": ["open"],
                "additionalProperties": False,
            },
            handler=(tools.set_driver_window),
            # This flag will become important when
            # SafetyPolicy is introduced.
            requires_confirmation=True,
        ),
    ]

from __future__ import annotations

from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolResult,
)


# ============================================================
# Climate Tools
# ============================================================


class ClimateTools:
    """
    Mock climate-control backend.

    Later this class can be replaced by Android Vehicle API,
    CAN signals, CARLA or OEM middleware without changing the
    Agent interface.
    """

    def __init__(
        self,
        context_manager: ContextManager,
    ):
        self.context_manager = (
            context_manager
        )

    # ========================================================
    # Get climate
    # ========================================================

    def get_climate_status(
        self,
    ) -> ToolResult:

        context = (
            self.context_manager
            .get_context()
        )

        vehicle = (
            context.vehicle
        )

        return ToolResult(
            success=True,
            message=(
                "Climate status retrieved."
            ),
            data={
                "cabin_temperature_c":
                    vehicle
                    .cabin_temperature_c,

                "target_temperature_c":
                    vehicle
                    .target_temperature_c,

                "ac_enabled":
                    vehicle
                    .ac_enabled,
            },
        )

    # ========================================================
    # Set temperature
    # ========================================================

    def set_temperature(
        self,
        temperature_c: float,
    ) -> ToolResult:

        temperature_c = float(
            temperature_c
        )

        if not (
            16.0
            <= temperature_c
            <= 30.0
        ):

            return ToolResult(
                success=False,
                message=(
                    "Requested temperature "
                    "is outside the supported "
                    "range."
                ),
                error=(
                    "TEMPERATURE_OUT_OF_RANGE"
                ),
                data={
                    "minimum_c": 16.0,
                    "maximum_c": 30.0,
                },
            )

        context_before = (
            self.context_manager
            .get_context()
        )

        old_temperature = (
            context_before
            .vehicle
            .target_temperature_c
        )

        self.context_manager.update_vehicle(
            target_temperature_c=(
                temperature_c
            ),
            ac_enabled=True,
        )

        return ToolResult(
            success=True,
            message=(
                "Target cabin temperature "
                f"set to {temperature_c:.1f}°C."
            ),
            data={
                "old_temperature_c":
                    old_temperature,

                "target_temperature_c":
                    temperature_c,

                "ac_enabled":
                    True,
            },
        )

    # ========================================================
    # Set AC
    # ========================================================

    def set_ac(
        self,
        enabled: bool,
    ) -> ToolResult:

        enabled = bool(
            enabled
        )

        self.context_manager.update_vehicle(
            ac_enabled=enabled
        )

        return ToolResult(
            success=True,
            message=(
                "Air conditioning "
                + (
                    "enabled."
                    if enabled
                    else "disabled."
                )
            ),
            data={
                "ac_enabled":
                    enabled,
            },
        )


# ============================================================
# Registration
# ============================================================


def build_climate_tools(
    context_manager: ContextManager,
) -> list[ToolDefinition]:

    tools = ClimateTools(
        context_manager
    )

    return [
        ToolDefinition(
            name="get_climate_status",
            description=(
                "Get the current cabin "
                "temperature, target "
                "temperature and air "
                "conditioning status."
            ),
            category="climate",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties":
                    False,
            },
            handler=(
                tools.get_climate_status
            ),
        ),

        ToolDefinition(
            name="set_temperature",
            description=(
                "Set the target cabin "
                "temperature in Celsius."
            ),
            category="climate",
            parameters={
                "type": "object",
                "properties": {
                    "temperature_c": {
                        "type": "number",
                        "minimum": 16,
                        "maximum": 30,
                        "description": (
                            "Target cabin "
                            "temperature in "
                            "degrees Celsius."
                        ),
                    },
                },
                "required": [
                    "temperature_c"
                ],
                "additionalProperties":
                    False,
            },
            handler=(
                tools.set_temperature
            ),
        ),

        ToolDefinition(
            name="set_ac",
            description=(
                "Turn the cabin air "
                "conditioning on or off."
            ),
            category="climate",
            parameters={
                "type": "object",
                "properties": {
                    "enabled": {
                        "type":
                            "boolean",
                    },
                },
                "required": [
                    "enabled"
                ],
                "additionalProperties":
                    False,
            },
            handler=(
                tools.set_ac
            ),
        ),
    ]

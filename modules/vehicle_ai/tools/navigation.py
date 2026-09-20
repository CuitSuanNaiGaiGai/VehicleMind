from __future__ import annotations

from modules.vehicle_ai.context import (
    ContextManager,
    NavigationState,
)

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolResult,
)


# ============================================================
# Mock POI
# ============================================================


MOCK_REST_AREAS = [
    {
        "name":
            "West Lake Rest Area",

        "distance_km":
            6.8,

        "eta_minutes":
            8,
    },

    {
        "name":
            "Riverside Service Area",

        "distance_km":
            12.4,

        "eta_minutes":
            15,
    },
]


# ============================================================
# Navigation Tools
# ============================================================


class NavigationTools:

    def __init__(
        self,
        context_manager: ContextManager,
    ):
        self.context_manager = (
            context_manager
        )

    # ========================================================
    # Search rest area
    # ========================================================

    def search_nearby_rest_area(
        self,
    ) -> ToolResult:
        """
        Mock implementation.

        Later replace with a real POI / map API.
        """

        result = (
            MOCK_REST_AREAS[0]
        )

        return ToolResult(
            success=True,
            message=(
                "Nearby rest area found."
            ),
            data={
                **result,
            },
        )

    # ========================================================
    # Start navigation
    # ========================================================

    def start_navigation(
        self,
        destination: str,
    ) -> ToolResult:

        destination = (
            destination.strip()
        )

        if not destination:

            return ToolResult(
                success=False,
                message=(
                    "Navigation destination "
                    "cannot be empty."
                ),
                error=(
                    "EMPTY_DESTINATION"
                ),
            )

        self.context_manager.update_vehicle(
            navigation_destination=(
                destination
            ),
            navigation_state=(
                NavigationState.ACTIVE
            ),
        )

        return ToolResult(
            success=True,
            message=(
                "Navigation started to "
                f"'{destination}'."
            ),
            data={
                "destination":
                    destination,

                "navigation_state":
                    NavigationState.ACTIVE,
            },
        )

    # ========================================================
    # Cancel
    # ========================================================

    def cancel_navigation(
        self,
    ) -> ToolResult:

        self.context_manager.update_vehicle(
            navigation_destination=None,
            navigation_state=(
                NavigationState.IDLE
            ),
        )

        return ToolResult(
            success=True,
            message=(
                "Navigation cancelled."
            ),
            data={
                "navigation_state":
                    NavigationState.IDLE,
            },
        )


# ============================================================
# Registration
# ============================================================


def build_navigation_tools(
    context_manager: ContextManager,
) -> list[ToolDefinition]:

    tools = NavigationTools(
        context_manager
    )

    return [
        ToolDefinition(
            name=(
                "search_nearby_rest_area"
            ),
            description=(
                "Find a nearby rest area "
                "or service area where "
                "the driver can stop and "
                "rest."
            ),
            category="navigation",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties":
                    False,
            },
            handler=(
                tools
                .search_nearby_rest_area
            ),
        ),

        ToolDefinition(
            name="start_navigation",
            description=(
                "Start navigation to "
                "a destination."
            ),
            category="navigation",
            parameters={
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": (
                            "Destination "
                            "name or address."
                        ),
                    },
                },
                "required": [
                    "destination"
                ],
                "additionalProperties":
                    False,
            },
            handler=(
                tools.start_navigation
            ),
            requires_confirmation=True,
        ),

        ToolDefinition(
            name="cancel_navigation",
            description=(
                "Cancel the active "
                "navigation session."
            ),
            category="navigation",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties":
                    False,
            },
            handler=(
                tools.cancel_navigation
            ),
            requires_confirmation=True,
        ),
    ]

from __future__ import annotations

from typing import Any

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolResult,
)


# ============================================================
# Tool Registry
# ============================================================


class ToolRegistry:
    """
    Central registry for VehicleMind tools.

    Future flow:

        LLM proposes:
            {
                "name": "set_temperature",
                "arguments": {
                    "temperature_c": 23
                }
            }

        ↓

        ToolRegistry.execute(...)

        ↓

        ToolResult
    """

    def __init__(
        self,
    ):
        self._tools: dict[
            str,
            ToolDefinition,
        ] = {}

    # ========================================================
    # Register
    # ========================================================

    def register(
        self,
        tool: ToolDefinition,
    ) -> None:

        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    # ========================================================
    # Lookup
    # ========================================================

    def get(
        self,
        name: str,
    ) -> ToolDefinition:

        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")

        return self._tools[name]

    def names(
        self,
    ) -> list[str]:

        return sorted(self._tools.keys())

    # ========================================================
    # Schemas
    # ========================================================

    def llm_schemas(
        self,
    ) -> list[dict[str, Any]]:
        """
        Schemas that can later be passed to a function-calling
        capable LLM.
        """

        return [tool.llm_schema() for tool in self._tools.values()]

    # ========================================================
    # Execute
    # ========================================================

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:

        if arguments is None:
            arguments = {}

        if name not in self._tools:
            return ToolResult(
                success=False,
                message=(f"Tool '{name}' is not available."),
                error="UNKNOWN_TOOL",
            )

        tool = self._tools[name]

        # ----------------------------------------------------
        # Basic required-field validation
        # ----------------------------------------------------

        required = tool.parameters.get(
            "required",
            [],
        )

        missing = [field_name for field_name in required if field_name not in arguments]

        if missing:
            return ToolResult(
                success=False,
                message=(f"Missing required arguments: {missing}"),
                error=("MISSING_ARGUMENTS"),
                data={
                    "missing": missing,
                },
            )

        # ----------------------------------------------------
        # Reject unknown arguments.
        # ----------------------------------------------------

        known_properties = tool.parameters.get(
            "properties",
            {},
        )

        unknown = [key for key in arguments if key not in known_properties]

        if unknown:
            return ToolResult(
                success=False,
                message=(f"Unknown arguments: {unknown}"),
                error=("UNKNOWN_ARGUMENTS"),
                data={
                    "unknown": unknown,
                },
            )

        # ----------------------------------------------------
        # Execute tool
        # ----------------------------------------------------

        try:
            result = tool.handler(**arguments)

        except Exception as exc:
            return ToolResult(
                success=False,
                message=(f"Tool '{name}' execution failed."),
                error=(type(exc).__name__),
                data={
                    "detail": str(exc),
                },
            )

        if not isinstance(
            result,
            ToolResult,
        ):
            raise TypeError(f"Tool '{name}' must return ToolResult.")

        return result

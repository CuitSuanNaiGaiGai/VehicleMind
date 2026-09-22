from __future__ import annotations

from copy import deepcopy
from typing import Protocol
from typing import Any
from collections.abc import Mapping

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolExecutionRecord,
    ToolResult,
)


class Confirmation(Protocol):
    @property
    def action_id(self) -> str: ...

    @property
    def tool_name(self) -> str: ...

    @property
    def arguments(self) -> Mapping[str, Any]: ...


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
        self._execution_history: list[ToolExecutionRecord] = []
        self._used_confirmation_ids: set[str] = set()

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

    def execution_history(self) -> tuple[ToolExecutionRecord, ...]:
        return tuple(deepcopy(self._execution_history))

    def _record(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any],
        requires_confirmation: bool,
        confirmed: bool,
        result: ToolResult,
    ) -> ToolResult:
        self._execution_history.append(
            ToolExecutionRecord(
                name=name,
                arguments=deepcopy(dict(arguments)),
                requires_confirmation=requires_confirmation,
                confirmed=confirmed,
                success=result.success,
                error=result.error,
            )
        )
        return result

    # ========================================================
    # Execute
    # ========================================================

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        confirmation: Confirmation | None = None,
    ) -> ToolResult:

        if arguments is None:
            arguments = {}

        if name not in self._tools:
            return self._record(
                name=name,
                arguments=arguments,
                requires_confirmation=False,
                confirmed=False,
                result=ToolResult(
                    success=False,
                    message=(f"Tool '{name}' is not available."),
                    error="UNKNOWN_TOOL",
                ),
            )

        tool = self._tools[name]
        confirmed = False

        if tool.requires_confirmation:
            if confirmation is None:
                return self._record(
                    name=name,
                    arguments=arguments,
                    requires_confirmation=True,
                    confirmed=False,
                    result=ToolResult(
                        success=False,
                        message="Explicit confirmation is required.",
                        error="CONFIRMATION_REQUIRED",
                    ),
                )
            if (
                confirmation.tool_name != name
                or dict(confirmation.arguments) != arguments
            ):
                return self._record(
                    name=name,
                    arguments=arguments,
                    requires_confirmation=True,
                    confirmed=False,
                    result=ToolResult(
                        success=False,
                        message="Confirmation does not match the pending action.",
                        error="CONFIRMATION_MISMATCH",
                    ),
                )
            if confirmation.action_id in self._used_confirmation_ids:
                return self._record(
                    name=name,
                    arguments=arguments,
                    requires_confirmation=True,
                    confirmed=False,
                    result=ToolResult(
                        success=False,
                        message="Confirmation was already used.",
                        error="CONFIRMATION_REPLAY",
                    ),
                )
            self._used_confirmation_ids.add(confirmation.action_id)
            confirmed = True

        # ----------------------------------------------------
        # Basic required-field validation
        # ----------------------------------------------------

        required = tool.parameters.get(
            "required",
            [],
        )

        missing = [field_name for field_name in required if field_name not in arguments]

        if missing:
            return self._record(
                name=name,
                arguments=arguments,
                requires_confirmation=tool.requires_confirmation,
                confirmed=confirmed,
                result=ToolResult(
                    success=False,
                    message=(f"Missing required arguments: {missing}"),
                    error=("MISSING_ARGUMENTS"),
                    data={
                        "missing": missing,
                    },
                ),
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
            return self._record(
                name=name,
                arguments=arguments,
                requires_confirmation=tool.requires_confirmation,
                confirmed=confirmed,
                result=ToolResult(
                    success=False,
                    message=(f"Unknown arguments: {unknown}"),
                    error=("UNKNOWN_ARGUMENTS"),
                    data={
                        "unknown": unknown,
                    },
                ),
            )

        # ----------------------------------------------------
        # Execute tool
        # ----------------------------------------------------

        try:
            result = tool.handler(**arguments)

        except Exception as exc:
            return self._record(
                name=name,
                arguments=arguments,
                requires_confirmation=tool.requires_confirmation,
                confirmed=confirmed,
                result=ToolResult(
                    success=False,
                    message=(f"Tool '{name}' execution failed."),
                    error=(type(exc).__name__),
                    data={
                        "detail": str(exc),
                    },
                ),
            )

        if not isinstance(
            result,
            ToolResult,
        ):
            raise TypeError(f"Tool '{name}' must return ToolResult.")

        return self._record(
            name=name,
            arguments=arguments,
            requires_confirmation=tool.requires_confirmation,
            confirmed=confirmed,
            result=result,
        )

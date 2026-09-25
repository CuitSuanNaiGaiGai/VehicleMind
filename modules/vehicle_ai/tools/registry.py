from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

from modules.vehicle_ai.tools.base import (
    ToolDefinition,
    ToolExecutionRecord,
    ToolResult,
)


@dataclass(frozen=True)
class _ConfirmationGrant:
    action_id: str
    tool_name: str
    arguments: Mapping[str, Any]


class ConfirmationIssuer:
    """Single-owner capability that can mint grants for one registry."""

    def __init__(self, registry: ToolRegistry, issuer_key: object) -> None:
        self.__registry = registry
        self.__issuer_key = issuer_key

    def issue(
        self,
        action_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> _ConfirmationGrant:
        return self.__registry._issue_confirmation(
            self.__issuer_key,
            action_id,
            tool_name,
            arguments,
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
        self._execution_history: list[ToolExecutionRecord] = []
        self._issued_confirmations: dict[str, _ConfirmationGrant] = {}
        self._used_confirmation_ids: set[str] = set()
        self.__issuer_key = object()
        self.__issuer_available = True

    def take_confirmation_issuer(self) -> ConfirmationIssuer:
        """Transfer the sole confirmation-issuing capability to a controller."""

        if not self.__issuer_available:
            raise RuntimeError("confirmation issuer has already been claimed")
        self.__issuer_available = False
        return ConfirmationIssuer(self, self.__issuer_key)

    def _issue_confirmation(
        self,
        issuer_key: object,
        action_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> _ConfirmationGrant:
        """Mint an identity-bound grant after the pending store consumes an action."""

        if issuer_key is not self.__issuer_key:
            raise PermissionError("invalid confirmation issuer capability")
        grant = _ConfirmationGrant(
            action_id=action_id,
            tool_name=tool_name,
            arguments=deepcopy(dict(arguments)),
        )
        self._issued_confirmations[action_id] = grant
        return grant

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
                result_data=deepcopy(result.data),
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
        confirmation: object | None = None,
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
            if not isinstance(confirmation, _ConfirmationGrant):
                return self._record(
                    name=name,
                    arguments=arguments,
                    requires_confirmation=True,
                    confirmed=False,
                    result=ToolResult(
                        success=False,
                        message="Confirmation was not issued by this registry.",
                        error="INVALID_CONFIRMATION",
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
            if (
                self._issued_confirmations.get(confirmation.action_id)
                is not confirmation
            ):
                return self._record(
                    name=name,
                    arguments=arguments,
                    requires_confirmation=True,
                    confirmed=False,
                    result=ToolResult(
                        success=False,
                        message="Confirmation is not live in this registry.",
                        error="INVALID_CONFIRMATION",
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
            self._issued_confirmations.pop(confirmation.action_id)
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
                        "outcome_unknown": not tool.read_only,
                    },
                ),
            )

        if not isinstance(
            result,
            ToolResult,
        ):
            result = ToolResult(
                False,
                "Tool returned an invalid result.",
                data={"outcome_unknown": not tool.read_only},
                error="INVALID_TOOL_RESULT",
            )

        return self._record(
            name=name,
            arguments=arguments,
            requires_confirmation=tool.requires_confirmation,
            confirmed=confirmed,
            result=result,
        )

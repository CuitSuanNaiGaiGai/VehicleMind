from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable


# ============================================================
# Tool Result
# ============================================================


@dataclass
class ToolResult:
    """
    Unified result returned by every VehicleMind tool.

    The future LLM never directly receives arbitrary Python
    objects. All tool executions return this structured form.
    """

    success: bool

    message: str

    data: dict[str, Any] = field(default_factory=dict)

    error: str | None = None
    policy: dict[str, Any] | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "policy": self.policy,
        }


@dataclass(frozen=True)
class ToolExecutionRecord:
    """Auditable outcome for one attempted tool execution."""

    name: str
    arguments: dict[str, Any]
    requires_confirmation: bool
    confirmed: bool
    success: bool
    error: str | None
    result_data: dict[str, Any] = field(default_factory=dict)
    user_intent: str = ""
    policy: dict[str, Any] | None = None


# ============================================================
# Tool Definition
# ============================================================


@dataclass
class ToolDefinition:
    """
    One callable vehicle capability.

    parameters follows JSON Schema so that this structure can
    later be directly exposed to an LLM Function Calling API.
    """

    name: str

    description: str

    parameters: dict[str, Any]

    handler: Callable[..., ToolResult]

    category: str = "general"

    requires_confirmation: bool = False
    read_only: bool = False

    def llm_schema(
        self,
    ) -> dict[str, Any]:
        """
        Provider-friendly function-calling schema.
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

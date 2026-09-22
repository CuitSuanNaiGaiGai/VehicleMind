from __future__ import annotations

from typing import Any

from modules.vehicle_ai.agent.action_state import PendingAction, PendingActionStore
from modules.vehicle_ai.tools import ToolRegistry, ToolResult


class ActionConfirmationController:
    """Stage and consume sensitive actions outside natural-language history."""

    def __init__(
        self,
        pending_actions: PendingActionStore,
        tool_registry: ToolRegistry,
    ) -> None:
        self.pending_actions = pending_actions
        self.tool_registry = tool_registry

    def stage(self, tool_name: str, arguments: dict[str, Any]) -> None:
        try:
            tool = self.tool_registry.get(tool_name)
        except KeyError:
            return
        if not tool.requires_confirmation:
            return

        pending = self.pending_actions.get()
        if (
            pending is not None
            and pending.tool_name == tool_name
            and pending.arguments == arguments
        ):
            return
        self.pending_actions.set(
            PendingAction(
                tool_name=tool_name,
                arguments=dict(arguments),
                display_text=f"Confirm vehicle action: {tool_name}",
            )
        )

    def confirm(self, action_id: str) -> ToolResult:
        confirmation = self.pending_actions.consume(action_id)
        if confirmation is None:
            return ToolResult(
                success=False,
                message="No matching live pending action was found.",
                error="INVALID_CONFIRMATION",
            )
        return self.tool_registry.execute(
            confirmation.tool_name,
            dict(confirmation.arguments),
            confirmation=confirmation,
        )

from __future__ import annotations

from typing import Any

from modules.vehicle_ai.agent.action_state import PendingAction, PendingActionStore
from modules.vehicle_ai.tools import ConfirmationIssuer, ToolRegistry, ToolResult


class ActionConfirmationController:
    """Stage and consume sensitive actions outside natural-language history."""

    def __init__(
        self,
        pending_actions: PendingActionStore,
        tool_registry: ToolRegistry,
        confirmation_issuer: ConfirmationIssuer,
    ) -> None:
        self.pending_actions = pending_actions
        self.tool_registry = tool_registry
        self.__confirmation_issuer = confirmation_issuer

    def stage(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_intent: str = "",
        *,
        display_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
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
                display_text=display_text or f"确认车机操作：{tool_name}",
                metadata={"user_intent": user_intent, **(metadata or {})},
                created_at=self.pending_actions.now(),
            )
        )

    def confirm(self, action_id: str) -> ToolResult:
        pending = self.pending_actions.get()
        user_intent = (
            str(pending.metadata.get("user_intent", ""))
            if pending is not None and pending.action_id == action_id
            else ""
        )
        confirmation = self.pending_actions.consume(action_id)
        if confirmation is None:
            return ToolResult(
                success=False,
                message="No matching live pending action was found.",
                error="INVALID_CONFIRMATION",
            )
        grant = self.__confirmation_issuer.issue(
            confirmation.action_id,
            confirmation.tool_name,
            confirmation.arguments,
        )
        return self.tool_registry.execute(
            confirmation.tool_name,
            dict(confirmation.arguments),
            confirmation=grant,
            user_intent=user_intent,
        )

    def reject(self, action_id: str) -> ToolResult:
        if not self.pending_actions.reject(action_id):
            return ToolResult(
                success=False,
                message="No matching live pending action was found.",
                error="INVALID_CONFIRMATION",
            )
        return ToolResult(success=True, message="Pending action cancelled.")

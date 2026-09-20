from __future__ import annotations

import time
import uuid

from dataclasses import dataclass
from dataclasses import field
from typing import Any


# ============================================================
# Pending Action
# ============================================================


@dataclass
class PendingAction:
    """
    A grounded action waiting for user confirmation.

    Example:

        search_nearby_rest_area()
                ↓
        rest_area_001
                ↓
        PendingAction(
            tool_name="start_navigation",
            arguments={
                "poi_id": "rest_area_001"
            }
        )

    Natural-language labels may change, but the canonical
    arguments remain stable.
    """

    tool_name: str

    arguments: dict[str, Any]

    display_text: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    expires_after_seconds: float = 120.0

    action_id: str = field(
        default_factory=lambda: (
            uuid.uuid4().hex
        )
    )

    # ========================================================
    # Lifetime
    # ========================================================

    def age_seconds(
        self,
        now: float | None = None,
    ) -> float:

        if now is None:
            now = time.time()

        return max(
            0.0,
            now - self.created_at,
        )

    def is_expired(
        self,
        now: float | None = None,
    ) -> bool:

        return (
            self.age_seconds(now)
            > self.expires_after_seconds
        )

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "action_id":
                self.action_id,

            "tool_name":
                self.tool_name,

            "arguments":
                self.arguments,

            "display_text":
                self.display_text,

            "metadata":
                self.metadata,

            "created_at":
                self.created_at,

            "expires_after_seconds":
                self.expires_after_seconds,
        }


# ============================================================
# Pending Action Store
# ============================================================


class PendingActionStore:
    """
    Session-level action state.

    The store intentionally lives outside normal conversation
    history.

    This prevents the execution layer from depending on an
    LLM paraphrasing an entity correctly.
    """

    def __init__(
        self,
    ):
        self._pending: (
            PendingAction
            | None
        ) = None

    # ========================================================
    # Set
    # ========================================================

    def set(
        self,
        action: PendingAction,
    ) -> None:

        self._pending = action

    # ========================================================
    # Get
    # ========================================================

    def get(
        self,
    ) -> PendingAction | None:

        if self._pending is None:
            return None

        if self._pending.is_expired():

            self._pending = None

            return None

        return self._pending

    # ========================================================
    # Clear
    # ========================================================

    def clear(
        self,
    ) -> None:

        self._pending = None

    # ========================================================
    # Agent representation
    # ========================================================

    def to_agent_context(
        self,
    ) -> dict[str, Any] | None:

        action = self.get()

        if action is None:
            return None

        return {
            "action_id":
                action.action_id,

            "tool_name":
                action.tool_name,

            "arguments":
                action.arguments,

            "display_text":
                action.display_text,

            "metadata":
                action.metadata,
        }

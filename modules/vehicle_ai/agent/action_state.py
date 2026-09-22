from __future__ import annotations

import time
import uuid

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return deepcopy(value)


# ============================================================
# Pending Action
# ============================================================


@dataclass(frozen=True)
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

    arguments: Mapping[str, Any]

    display_text: str

    metadata: Mapping[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=time.time)

    expires_after_seconds: float = 120.0

    action_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_value(self.arguments))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

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

        return self.age_seconds(now) > self.expires_after_seconds

    # ========================================================
    # Serialization
    # ========================================================

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "arguments": _plain_value(self.arguments),
            "display_text": self.display_text,
            "metadata": _plain_value(self.metadata),
            "created_at": self.created_at,
            "expires_after_seconds": self.expires_after_seconds,
        }


@dataclass(frozen=True)
class ConfirmedAction:
    """One immutable, one-time grant derived from a live pending action."""

    action_id: str
    tool_name: str
    arguments: Mapping[str, Any]


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
        self._pending: PendingAction | None = None

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

    def consume(
        self,
        action_id: str,
    ) -> ConfirmedAction | None:
        """Consume one matching live action and return its exact execution grant."""

        action = self.get()
        if action is None or action.action_id != action_id:
            return None
        self.clear()
        return ConfirmedAction(
            action_id=action.action_id,
            tool_name=action.tool_name,
            arguments=_freeze_value(action.arguments),
        )

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
            "action_id": action.action_id,
            "tool_name": action.tool_name,
            "arguments": _plain_value(action.arguments),
            "display_text": action.display_text,
            "metadata": _plain_value(action.metadata),
        }

from __future__ import annotations

import time
import uuid

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import Any


# ============================================================
# Event Type
# ============================================================


class EventType(StrEnum):
    """
    Semantic events inside VehicleMind.

    These events are intentionally higher-level than raw
    ContextChange objects.
    """

    # --------------------------------------------------------
    # Driver
    # --------------------------------------------------------

    DRIVER_STATE_CHANGED = "DRIVER_STATE_CHANGED"

    DRIVER_RISK_CHANGED = "DRIVER_RISK_CHANGED"

    DRIVER_ABSENT = "DRIVER_ABSENT"

    DRIVER_PRESENT = "DRIVER_PRESENT"

    HIGH_RISK_DETECTED = "HIGH_RISK_DETECTED"

    # --------------------------------------------------------
    # Driving scene
    # --------------------------------------------------------

    TRAFFIC_LEVEL_CHANGED = "TRAFFIC_LEVEL_CHANGED"

    LANE_LOST = "LANE_LOST"

    DRIVABLE_AREA_LOST = "DRIVABLE_AREA_LOST"

    # --------------------------------------------------------
    # Vehicle
    # --------------------------------------------------------

    NAVIGATION_STATE_CHANGED = "NAVIGATION_STATE_CHANGED"

    # --------------------------------------------------------
    # Human interaction
    # --------------------------------------------------------

    USER_UTTERANCE = "USER_UTTERANCE"


# ============================================================
# Event priority
# ============================================================


class EventPriority(StrEnum):
    """
    Event priority.

    LOW
        Logging / passive information.

    MEDIUM
        Useful context change.

    HIGH
        Important event which may affect interaction.

    CRITICAL
        Safety-relevant event that deserves immediate
        attention from upper layers.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================
# Vehicle Event
# ============================================================


@dataclass(frozen=True)
class VehicleEvent:
    """
    Unified semantic event representation.
    """

    type: EventType

    priority: EventPriority

    source: str

    message: str

    data: dict[str, Any] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "event_id": self.event_id,
            "type": self.type,
            "priority": self.priority,
            "source": self.source,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def __str__(
        self,
    ) -> str:

        return f"[{self.priority}] {self.type}: {self.message}"


# ============================================================
# User event helper
# ============================================================


def create_user_utterance_event(
    text: str,
) -> VehicleEvent:
    """
    Convert one user message into an EventBus event.

    Later ASR output will use the same interface.
    """

    text = text.strip()

    if not text:
        raise ValueError("User utterance cannot be empty.")

    return VehicleEvent(
        type=(EventType.USER_UTTERANCE),
        priority=(EventPriority.HIGH),
        source="user",
        message=text,
        data={
            "text": text,
        },
    )

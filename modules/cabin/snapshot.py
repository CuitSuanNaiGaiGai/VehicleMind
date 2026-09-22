from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.observation import ObservationMetadata


@dataclass
class CabinPerceptionSnapshot:
    """Semantic cabin result and the quality of its source frame."""

    metadata: ObservationMetadata
    face_visible: bool
    presence: Any
    driver_state: Any
    risk: Any
    perclos: float | None
    perclos_ready: bool
    eye_closed: bool | None
    eye_closure_seconds: float
    recent_yawns: int
    blink_count: int
    current_yawn: bool

    @property
    def timestamp_ms(self) -> int:
        return self.metadata.timestamp_ms

    def to_context_kwargs(self) -> dict[str, Any]:
        return {
            "presence": self.presence,
            "driver_state": self.driver_state,
            "risk": self.risk,
            "perclos": self.perclos,
            "eye_closed": self.eye_closed,
            "eye_closure_seconds": self.eye_closure_seconds,
            "recent_yawns": self.recent_yawns,
            "blink_count": self.blink_count,
        }

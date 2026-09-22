from __future__ import annotations

import math

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class ObservationMetadata:
    """Quality and timing of one successful perception result."""

    timestamp_ms: int
    sequence: int
    source: str
    confidence: float | None
    valid: bool
    processing_ms: float

    def __post_init__(self) -> None:
        for name in ("timestamp_ms", "sequence"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be nonempty")
        if type(self.valid) is not bool:
            raise TypeError("valid must be a boolean")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, int | float
            ):
                raise TypeError("confidence must be numeric or null")
            if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be finite and within [0, 1]")
        if isinstance(self.processing_ms, bool) or not isinstance(
            self.processing_ms, int | float
        ):
            raise TypeError("processing_ms must be numeric")
        if not math.isfinite(self.processing_ms) or self.processing_ms < 0:
            raise ValueError("processing_ms must be finite and nonnegative")


class ObservationSequencer:
    """Assign consecutive IDs only after metadata validation succeeds."""

    def __init__(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be nonempty")
        self._source = source
        self._next_sequence = 0
        self._lock = Lock()

    def next(
        self,
        *,
        timestamp_ms: int,
        processing_ms: float,
        confidence: float | None = None,
        valid: bool = True,
    ) -> ObservationMetadata:
        with self._lock:
            metadata = ObservationMetadata(
                timestamp_ms=timestamp_ms,
                sequence=self._next_sequence,
                source=self._source,
                confidence=confidence,
                valid=valid,
                processing_ms=processing_ms,
            )
            self._next_sequence += 1
            return metadata

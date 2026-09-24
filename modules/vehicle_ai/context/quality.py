from __future__ import annotations

import time

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from collections.abc import Callable

from modules.observation import ObservationMetadata


class QualityStatus(StrEnum):
    MISSING = "MISSING"
    INVALID = "INVALID"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"
    KNOWN = "KNOWN"


@dataclass(frozen=True)
class Receipt:
    received_at: float
    valid: bool
    metadata: ObservationMetadata | None


class ObservationQualityTracker:
    """Track observation receipt separately from semantic context defaults."""

    _MAX_AGE = {"driver": 2.0, "road": 1.0, "vehicle": 2.0}

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._receipts: dict[str, Receipt] = {}
        self._field_receipts: dict[tuple[str, str], Receipt] = {}
        self._clock = clock

    def record(
        self,
        domain: str,
        *,
        valid: bool,
        metadata: ObservationMetadata | None = None,
        fields: tuple[str, ...] = (),
    ) -> None:
        if domain not in self._MAX_AGE:
            raise ValueError(f"unknown context domain: {domain}")
        receipt = Receipt(self._clock(), valid, metadata)
        self._receipts[domain] = receipt
        if valid:
            for field in fields:
                self._field_receipts[(domain, field)] = receipt

    def report(self, domain: str, *, now: float | None = None) -> dict[str, Any]:
        if domain not in self._MAX_AGE:
            raise ValueError(f"unknown context domain: {domain}")
        receipt = self._receipts.get(domain)
        if receipt is None:
            return {
                "status": QualityStatus.MISSING,
                "received_at": None,
                "age_seconds": None,
                "metadata": None,
            }
        current = self._clock() if now is None else now
        age = max(0.0, current - receipt.received_at)
        if not receipt.valid:
            status = QualityStatus.INVALID
        elif age > self._MAX_AGE[domain]:
            status = QualityStatus.STALE
        else:
            status = QualityStatus.KNOWN
        return {
            "status": status,
            "received_at": receipt.received_at,
            "age_seconds": age,
            "metadata": receipt.metadata,
        }

    def field_status(
        self, domain: str, field: str, value: object, *, now: float | None = None
    ) -> QualityStatus:
        domain_report = self.report(domain, now=now)
        if domain_report["status"] is QualityStatus.INVALID:
            return QualityStatus.INVALID
        receipt = self._field_receipts.get((domain, field))
        if receipt is None:
            return QualityStatus.MISSING
        current = self._clock() if now is None else now
        if current - receipt.received_at > self._MAX_AGE[domain]:
            return QualityStatus.STALE
        if value is None or value == "UNKNOWN":
            return QualityStatus.UNKNOWN
        return QualityStatus.KNOWN

    def field_evidence(self, domain: str, field: str) -> dict[str, Any]:
        """Return provenance for this field, not the domain's latest update."""
        receipt = self._field_receipts.get((domain, field))
        metadata = receipt.metadata if receipt else None
        return {
            "source": metadata.source if metadata else None,
            "timestamp_ms": metadata.timestamp_ms if metadata else None,
            "confidence": metadata.confidence if metadata else None,
            "age_seconds": (
                max(0.0, self._clock() - receipt.received_at) if receipt else None
            ),
        }

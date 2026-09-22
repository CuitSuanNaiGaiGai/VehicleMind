from __future__ import annotations

import time

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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

    def __init__(self) -> None:
        self._receipts: dict[str, Receipt] = {}

    def record(
        self,
        domain: str,
        *,
        valid: bool,
        metadata: ObservationMetadata | None = None,
    ) -> None:
        if domain not in self._MAX_AGE:
            raise ValueError(f"unknown context domain: {domain}")
        self._receipts[domain] = Receipt(time.monotonic(), valid, metadata)

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
        current = time.monotonic() if now is None else now
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


def field_quality(domain_status: QualityStatus, value: object) -> QualityStatus:
    if domain_status is not QualityStatus.KNOWN:
        return domain_status
    if value is None or value == "UNKNOWN":
        return QualityStatus.UNKNOWN
    return QualityStatus.KNOWN

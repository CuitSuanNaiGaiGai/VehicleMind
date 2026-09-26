"""Performance timing summaries for perception audit runs."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def summarize_ms(values: Sequence[float]) -> dict[str, float | int] | None:
    """Summarize non-negative timing samples in milliseconds."""
    if not values:
        return None

    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("timing samples must be finite and non-negative")

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }

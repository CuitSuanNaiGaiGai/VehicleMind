"""Small, deterministic statistics for per-frame benchmark timings."""

from __future__ import annotations

import math
import statistics

from collections.abc import Sequence


def _linear_percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize non-negative millisecond samples with linear percentiles."""

    if not values:
        raise ValueError("timing samples must not be empty")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("timing samples must be finite and non-negative")
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": _linear_percentile(ordered, 0.5),
        "p95": _linear_percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def throughput_fps(frames: int, elapsed_seconds: float) -> float:
    """Compute throughput from measured frames and wall-clock seconds."""

    if isinstance(frames, bool) or frames <= 0:
        raise ValueError("frame count must be positive")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed seconds must be finite and positive")
    return frames / elapsed_seconds

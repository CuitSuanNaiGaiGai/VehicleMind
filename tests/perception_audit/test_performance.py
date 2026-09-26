from __future__ import annotations

import math

import pytest

from modules.perception_audit.performance import summarize_ms


def test_empty_samples_return_none():
    assert summarize_ms([]) is None


def test_summary_uses_linear_interpolation_for_percentiles():
    summary = summarize_ms([10, 20, 40])

    assert summary == {
        "count": 3,
        "mean": 70 / 3,
        "p50": 20,
        "p95": 38,
        "min": 10,
        "max": 40,
    }


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, -math.inf])
def test_invalid_samples_are_rejected(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        summarize_ms([10, value])

from __future__ import annotations

import pytest

from modules.cabin.fatigue.blink import BlinkDetector
from modules.cabin.fatigue.perclos import PerclosEstimator
from modules.cabin.fatigue.yawn import YawnDetector


def test_blink_is_counted_only_after_valid_closed_duration() -> None:
    detector = BlinkDetector(min_closed_frames=2, max_closed_frames=3)

    detector.update(True)
    detector.update(True)
    result = detector.update(False)

    assert result.is_blinking is True
    assert result.blink_count == 1
    assert result.closed_frames == 0


def test_long_eye_closure_is_not_miscounted_as_blink() -> None:
    detector = BlinkDetector(min_closed_frames=2, max_closed_frames=3)

    for _ in range(4):
        detector.update(True)
    result = detector.update(False)

    assert result.is_blinking is False
    assert result.blink_count == 0


def test_perclos_ignores_missing_eye_observations() -> None:
    estimator = PerclosEstimator(
        window_seconds=10.0,
        min_observation_seconds=2.0,
    )

    estimator.update(0, True)
    estimator.update(1_000, True)
    estimator.update(2_000, None)
    result = estimator.update(3_000, False)

    assert result.observed_duration == pytest.approx(2.0)
    assert result.closed_duration == pytest.approx(2.0)
    assert result.perclos == pytest.approx(1.0)
    assert result.ready is True


def test_perclos_rejects_non_monotonic_timestamps() -> None:
    estimator = PerclosEstimator()
    estimator.update(100, False)

    with pytest.raises(ValueError, match="monotonically increasing"):
        estimator.update(100, True)


def test_yawn_event_fires_once_per_mouth_open_interval() -> None:
    detector = YawnDetector(min_open_seconds=1.2)

    detector.update(0, True)
    first = detector.update(1_200, True)
    repeated = detector.update(1_500, True)
    detector.update(1_600, False)
    second = detector.update(3_000, True)
    second = detector.update(4_200, True)

    assert first.yawn_event is True
    assert repeated.yawn_event is False
    assert second.yawn_event is True
    assert second.yawn_count == 2

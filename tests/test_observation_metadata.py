from __future__ import annotations

import pytest

from modules.observation import ObservationMetadata, ObservationSequencer


def test_metadata_keeps_unknown_confidence_distinct_from_validity() -> None:
    observations = ObservationSequencer("cabin_perception")

    first = observations.next(timestamp_ms=10, processing_ms=1.5)
    second = observations.next(timestamp_ms=20, processing_ms=2.0)

    assert first == ObservationMetadata(
        timestamp_ms=10,
        sequence=0,
        source="cabin_perception",
        confidence=None,
        valid=True,
        processing_ms=1.5,
    )
    assert second.sequence == 1
    assert second.confidence is None


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"timestamp_ms": -1}, ValueError),
        ({"sequence": -1}, ValueError),
        ({"source": ""}, ValueError),
        ({"confidence": 1.1}, ValueError),
        ({"confidence": float("nan")}, ValueError),
        ({"confidence": 10**400}, ValueError),
        ({"valid": 1}, TypeError),
        ({"processing_ms": -0.1}, ValueError),
        ({"processing_ms": float("inf")}, ValueError),
        ({"processing_ms": 10**400}, ValueError),
    ],
)
def test_metadata_rejects_invalid_fields(
    values: dict[str, object], error: type[Exception]
) -> None:
    input_values: dict[str, object] = {
        "timestamp_ms": 0,
        "sequence": 0,
        "source": "road_perception",
        "confidence": None,
        "valid": True,
        "processing_ms": 0.0,
    }
    input_values.update(values)

    with pytest.raises(error):
        ObservationMetadata(**input_values)


def test_rejected_observation_does_not_consume_sequence() -> None:
    observations = ObservationSequencer("road_perception")

    with pytest.raises(ValueError):
        observations.next(timestamp_ms=-1, processing_ms=0.1)

    assert observations.next(timestamp_ms=0, processing_ms=0.1).sequence == 0

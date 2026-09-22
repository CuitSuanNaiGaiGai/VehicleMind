# Perception Observation Metadata Design

## Scope

This increment addresses the second item of section 3 in `todolist.md`: every successful cabin and road perception result carries a source-timeline timestamp, sequence number, source, confidence, validity, and total processing latency. Context freshness and invalid-observation handling are the next increment.

## Contract

`modules/observation.py` provides an immutable `ObservationMetadata` and a per-service `ObservationSequencer`. Timestamp is milliseconds on the source's frame timeline. Sequence starts at zero and advances only when a result is produced. Processing latency is nonnegative milliseconds measured with a monotonic clock. Confidence is `None` for the current aggregate cabin and YOLOPv2 scene outputs because no calibrated aggregate confidence is available. A produced result has `valid=True`; failed inference raises and does not create a result. This differs from an `UNKNOWN` semantic state or `eye_closed=None`, both of which may occur in a valid result.

## Integration

`CabinPerceptionSnapshot` and `DrivingPerceptionSnapshot` expose `metadata`. The cabin snapshot type moves to a lightweight module so its contract is testable without the optional MediaPipe runtime; the service continues to expose the same type. The cabin snapshot keeps its existing `timestamp_ms` accessor for callers. Both services create metadata after successful processing. The existing `to_context_kwargs()` return shape remains unchanged until the context-quality increment, preserving current Replay and Agent behavior.

## Verification

Tests cover metadata type/range checks, monotonic per-service sequences, valid snapshots with unknown confidence, and a road service frame run using an injected fake detector. Full offline tests and source-size limits must pass.

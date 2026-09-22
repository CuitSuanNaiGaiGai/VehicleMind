# Perception Observation Metadata Implementation Plan

> **For agentic workers:** Execute task steps in order and record verified completion.

**Goal:** Add six consistent metadata fields to cabin and road perception results.

**Architecture:** One shared immutable metadata model and one per-service sequence generator. Perception services measure and attach metadata after each successful frame.

**Tech Stack:** Python 3.13 dataclasses, OpenCV/NumPy for the road smoke test, pytest, Ruff, mypy.

## Global Constraints

- Do not require model weights for unit tests.
- Do not invent an aggregate confidence score.
- Production modules stay below 500 lines.
- Commit and push each completed task.

### Task 1: Metadata contract and service integration

**Files:** `modules/observation.py`, `modules/cabin/snapshot.py`, `modules/cabin/perception_service.py`, `modules/driving/perception_service.py`, `tests/test_observation_metadata.py`, `tests/driving/test_perception_snapshot_metadata.py`, `todolist.md`.

**Interface:** `ObservationMetadata(timestamp_ms, sequence, source, confidence, valid, processing_ms)`; `ObservationSequencer(source).next(timestamp_ms, processing_ms, confidence=None, valid=True) -> ObservationMetadata`.

- [x] Write tests for invalid metadata, sequence increments, cabin snapshot metadata, and a road frame result using an injected detector.
- [x] Run focused tests and observe failures due to missing metadata contract.
- [x] Implement immutable metadata, per-service sequencers, and snapshot fields.
- [x] Run focused tests, full offline tests, Ruff, mypy, and source-size checks.
- [x] Update the evidence-backed checkbox in section 3 of `todolist.md`.
- [ ] Commit, push a feature branch, open a PR, and verify CI.

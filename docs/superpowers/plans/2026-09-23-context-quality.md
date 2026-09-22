# Runtime Context Quality Implementation Plan

**Goal:** Distinguish missing, invalid, stale, unknown, and known context without conflating `False` with missing evidence.

**Architecture:** A small quality tracker stores per-domain receipt times and validity; `ContextManager` records quality after successful semantic updates and exposes a field-status report. Runtime accepts optional observation metadata and short-circuits invalid observations.

**Tech Stack:** Python dataclasses and enum, monotonic clock, pytest.

## Global Constraints

- Preserve existing semantic context fields and update API behavior.
- Do not compare prerecorded media timestamps across clock domains.
- Keep production files below 500 lines.
- Commit and push the verified increment.

### Task 1: Quality model and context integration

**Files:** `modules/vehicle_ai/context/quality.py`, `modules/vehicle_ai/context/context_manager.py`, `modules/vehicle_ai/runtime.py`, `tests/vehicle_ai/test_context_quality.py`, `todolist.md`.

- [x] Write failing tests for initial missing, observed `False`, `UNKNOWN`, invalid observation, stale TTL, and failed update not changing quality.
- [x] Implement the tracker and report API with a monotonic clock.
- [x] Wire optional observation metadata through runtime updates without changing replay behavior.
- [x] Run focused and full offline tests, lint, type checks, and source-size policy.
- [ ] Mark evidence-backed checklist items, review the change, commit, push, and merge after CI.

# Temporal Hazard Events Implementation Plan

**Goal:** Prevent single-frame cabin/road hazards from generating repeated safety alerts while retaining deterministic offline replay.

**Architecture:** A reusable hazard gate tracks hold and cooldown on an explicit observation clock. The event detector continues emitting informational transitions and evaluates hazard gates on every valid cabin/road update, including unchanged values. A versioned YAML file defines demo timing.

**Tech Stack:** Python, PyYAML, pytest.

## Global Constraints

- No external model or camera in tests.
- Preserve existing event types and replay trace schema.
- Keep production files under 500 lines.
- Commit and push after verification.

### Task 1: Gate, integration, and replay

**Files:** `modules/vehicle_ai/events/temporal_gate.py`, `modules/config/events.py`, `modules/config/events.yaml`, `modules/vehicle_ai/events/event_detector.py`, `modules/vehicle_ai/runtime.py`, `modules/vehicle_ai/replay/runner.py`, `assets/scenarios/drowsy_rest_stop.yaml`, tests, `todolist.md`.

- [x] Write failing tests for hold, cooldown, recovery, startup defaults, and replay.
- [x] Implement gate and validated versioned timing configuration.
- [x] Evaluate gates on every valid observation using explicit replay time.
- [x] Run full offline tests and quality checks; update the checked task.
- [ ] Review, commit, push, verify CI, and merge.

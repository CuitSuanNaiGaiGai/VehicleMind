# Offline Video Pipeline Implementation Plan

**Goal:** Run optional real cabin/road models from offline videos through bounded, observable, independently scheduled stages.

**Architecture:** A reusable bounded channel and four-stage threaded pipeline own capture, inference, context update, and display. Video adapters provide monotonic source-frame timestamps; the existing integrated CLI starts cabin and road pipelines while the Agent remains user-triggered.

**Tech Stack:** Python threading, OpenCV, pytest.

## Global Constraints

- Fresh-clone CI requires no ignored videos or model weights.
- Local real-model smoke tests are optional and must report missing dependencies/assets clearly.
- No production file exceeds 500 lines.
- Each complete increment is committed and pushed through a PR.

### Task 1: Bounded channels and stage runner

**Files:** `modules/vehicle_ai/pipeline/channel.py`, `modules/vehicle_ai/pipeline/runner.py`, tests.

- [ ] Write failing tests for bounded drop-oldest and blocking policies, close/wake behavior, independent stage rates, shutdown, and errors.
- [ ] Implement the reusable channel and pipeline runner.
- [ ] Run focused tests, full offline suite, Ruff, mypy, and source-size checks.

### Task 2: Offline video integration

**Files:** `apps/vehicle_ai_demo/video_source.py`, `apps/vehicle_ai_demo/integrated_workers.py`, `apps/vehicle_ai_demo/integrated_demo.py`, tests, `todolist.md`.

- [ ] Write failing tests for local video timestamps across loops and option validation.
- [ ] Integrate separate cabin and road video/model pipelines with terminal display and health.
- [ ] Verify local video decode and optional real-model smoke test; document dependency/asset limits.
- [ ] Review, commit, push, verify CI, merge, and update task progress.

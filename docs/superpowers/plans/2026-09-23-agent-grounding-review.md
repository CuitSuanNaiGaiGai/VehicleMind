# Agent Grounding Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Give each pilot case explicit fact and inference boundaries, then produce blind review packets and validated human decisions without treating model output as gold.

**Architecture:** Rubrics live beside candidate scenarios as separate YAML files so evaluation inputs remain unchanged. A small loader validates source references against scenario observations and expected tools. A review module exports model-anonymous packets and validates independently supplied decisions; the existing deterministic grader remains conservative (`needs_review`).

**Tech Stack:** Python 3.13, PyYAML, pytest, existing `EvaluationCase` and `TrialResult`.

## Global Constraints

- Eight current pilot cases remain `candidate`; no automatic promotion to reviewed gold.
- Never include provider/model identity or API keys in blind review packets.
- Do not use the evaluated models to judge their own output.
- Preserve the current confirmation gate and trial trace format.

---

### Task 1: Rubric contract and eight candidate labels

**Files:** Create `modules/vehicle_ai/evaluation/rubric.py`, `scenarios/agent_eval/rubrics/*.yaml`; test `tests/vehicle_ai/evaluation/test_rubric.py`.

**Interfaces:** `load_rubric(path: Path, case: EvaluationCase) -> GroundingRubric` rejects mismatched IDs, unknown fields, missing source facts, and tool references not present in the case expectation.

- [ ] Write tests asserting all eight rubrics load and a mismatched fact value fails.
- [ ] Run the focused tests and verify they fail for the missing loader.
- [ ] Implement strict YAML validation and eight candidate rubrics with source facts, necessary answer claims, allowed inferences and forbidden inferences.
- [ ] Run focused tests and confirm all pass.

### Task 2: Blind packet and decision validation

**Files:** Create `modules/vehicle_ai/evaluation/review.py`, `modules/vehicle_ai/evaluation/review_cli.py`; test `tests/vehicle_ai/evaluation/test_review.py`.

**Interfaces:** `build_blind_packet(case, rubric, trial) -> dict` omits provider/model and request metadata. `validate_decision(packet, decision) -> dict` requires reviewer, verdict, evidence notes and a disposition for every necessary claim.

- [ ] Write tests for anonymity, complete fact decisions and rejection of a forged success with missing evidence.
- [ ] Run tests and verify they fail for the missing review module.
- [ ] Implement packet export and decision validation, with CLI reading existing local trial JSON.
- [ ] Run focused tests and confirm all pass.

### Task 3: Integration and checkpoint

**Files:** Update `scenarios/agent_eval/README.md`, `todolist.md`; create Chinese review guide under `docs/reports/`.

- [ ] Generate a blind packet from a real existing pilot trace and inspect it for model identity and secret leakage.
- [ ] Run full pytest, Ruff, mypy and `git diff --check`.
- [ ] Request code review, address important findings, commit and push this checkpoint.
- [ ] Stop and report completed work plus the next 40-case annotation task.

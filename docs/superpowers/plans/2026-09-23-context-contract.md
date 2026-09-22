# Versioned Context Contract Implementation Plan

> **For agentic workers:** Execute this plan task by task. Each step is checked when verified.

**Goal:** Make Driver/Road/Vehicle updates validated and atomic under one versioned field contract.

**Architecture:** `contract.py` owns field rules and version. `ContextManager` validates before replacing a domain snapshot. Serialization exposes the version; replay reuses the version constant.

**Tech Stack:** Python 3.13 dataclasses, pytest, Ruff, mypy.

## Global Constraints

- Production modules stay below 500 lines.
- No perception accuracy claims or model execution are added.
- Every completed task is committed and pushed to GitHub.

### Task 1: Contract and atomic updates

**Files:** `modules/vehicle_ai/context/contract.py`, `context_manager.py`, `models.py`, `__init__.py`, `modules/vehicle_ai/replay/trace.py`, `tests/vehicle_ai/test_context_contract.py`, `todolist.md`.

**Interface:** `validate_domain_updates(domain: str, updates: Mapping[str, object]) -> None`; `CONTEXT_SCHEMA_VERSION: int`; `CONTEXT_FIELD_CONTRACTS: Mapping[str, Mapping[str, FieldContract]]`.

- [x] Write failing tests for complete field coverage, version serialization, invalid values, and rollback after a later field fails.
- [x] Run `uv run --group dev python -m pytest tests/vehicle_ai/test_context_contract.py -q` and confirm the expected failures.
- [x] Implement the field contract, validated atomic domain replacement, and shared version constant.
- [x] Run focused tests, replay regression, Ruff, mypy, source-size check, then full offline pytest.
- [x] Update only evidence-backed section 3 checkboxes in `todolist.md`.
- [x] Commit, push a feature branch, open a PR, and verify CI.

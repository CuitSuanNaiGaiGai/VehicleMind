# Agent PendingAction Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make refusal, target change, and expiry invalidate the old sensitive action before any Agent or tool execution.

**Architecture:** Keep the one-action store as the authority. A small deterministic intent classifier runs before LLM construction; explicit rejection returns immediately, while target-change clears the old action and continues through normal search and tool gating. Confirmation remains action-ID-bound in the controller and ToolRegistry.

**Tech Stack:** Python 3.13, pytest, Ruff, mypy, scripted offline LLM.

## Global Constraints

- No online LLM, video, camera, or model weight is required for tests.
- No natural-language text may mint a ToolRegistry confirmation grant.
- New code should remain in focused files; `vehicle_agent.py` stays under the 500-line policy.
- A cleared or expired action ID must never execute a sensitive tool.

---

### Task 1: Action-state rejection and expiry boundary

**Files:** Modify `modules/vehicle_ai/agent/action_state.py` and `modules/vehicle_ai/agent/confirmation.py`; test `tests/vehicle_ai/test_tool_confirmation.py`.

**Interfaces:** `PendingActionStore.reject(action_id: str) -> bool`; `VehicleAgent.reject_pending(action_id: str) -> ToolResult`; `PendingAction.is_expired(now: float | None = None) -> bool`.

- [ ] Add tests showing wrong ID preserves pending, correct ID clears it and makes later confirmation return `INVALID_CONFIRMATION`, and `created_at + TTL` is already expired.
- [ ] Run the focused tests and confirm they fail for the missing behavior.
- [ ] Implement `reject` using `get()` plus ID equality and `clear()`, expose a structured controller result, and change expiry comparison to `>=`.
- [ ] Run focused and full confirmation tests; commit the independently testable state change.

### Task 2: Deterministic refusal and target-change entry

**Files:** Create `modules/vehicle_ai/agent/pending_intent.py`; modify `modules/vehicle_ai/agent/vehicle_agent.py`; test `tests/vehicle_ai/test_pending_intent.py` and `tests/vehicle_ai/test_tool_confirmation.py`.

**Interfaces:** `classify_pending_intent(text: str) -> Literal["reject", "change_target"] | None`; `VehicleAgent.chat(user_text: str, debug: bool = True) -> str`.

- [ ] Test exact, normalized refusal expressions (`取消`, `不要了`, `cancel`, `no thanks`) and target-change prefixes (`换成`, `改去`, `instead`, `change destination to`); ambiguous text yields `None`.
- [ ] Confirm classifier tests fail, then add a small anchored classifier that does not interpret arbitrary text or produce confirmation.
- [ ] Test that refusal clears pending without calling LLM or executing tools; target-change clears old pending before the LLM sees context and no longer forces the old `poi_id` into a new tool call.
- [ ] Confirm integration tests fail, then wire the classifier at the start of `chat`; run focused tests and commit.

### Task 3: Regression evidence and checklist

**Files:** Modify `tests/vehicle_ai/test_tool_confirmation.py`, `todolist.md`, and a concise Agent safety note in `README.md` or `docs/`.

**Interfaces:** Existing `confirm_pending(action_id)` and ToolRegistry execution-history records.

- [ ] Add scripted tests for old-ID replay after target replacement, timeout without confirmation, and a hostile LLM sensitive tool call after rejection. Assert zero successful unconfirmed sensitive executions.
- [ ] Run targeted tests and fix only demonstrated failures.
- [ ] Run `uv run --group dev pytest -q`, `ruff check apps modules scripts tests`, `mypy` on changed modules, and `scripts/check_source_size.py`.
- [ ] Mark only evidenced todolist items complete, request independent code review, commit, push, create PR, await CI, merge, and sync local `main`.

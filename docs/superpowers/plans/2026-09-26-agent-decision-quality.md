# VehicleMind Agent Decision Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. The user has already chosen GPT-6 Luna Max implementation and GPT-6 Sol review; do not ask for an execution-mode choice. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复已复现的跨域事实缺失与目的地变更错误，并让中文 Demo 展示模型实际获得的依据和真实工具结果。

**Architecture:** 沿用现有 ContextSelector、质量契约、TaskPlanFlow 与确认门；增设小型决策简报和精确目标解析模块。模型获得本轮必需事实与未知原因，改目标搜索后的用户回复由本次真实候选及 pending 状态生成。使用已有回放、在线 runner、冻结审核器和 HTML 报告，不增加模型回合或框架依赖。

**Tech Stack:** Python 3.13, dataclasses, existing pytest/ruff/mypy, YAML cases, Qwen/GLM adapters, existing HTML report renderer.

## Global Constraints

- Execute in the isolated worktree on branch `codex/agent-decision-quality`, base `1d38e06`; the parent supplies the local workspace path separately. Keep the original checkout untouched.
- Run `python -m ...` from the isolated worktree, using the interpreter supplied by the parent; do not invoke `bin/pytest`, which may import the original checkout through editable installation.
- Use the parent's supplied `UV_PYTHON` and temporary `UV_CACHE_DIR` environment for packaging checks to find installed hatchling. The verified baseline is 674 passing tests after this environment correction.
- User has authorized implementation, relevant tests, commits, push/merge and configured API usage; no routine confirmation is necessary. Worktree writes may require tool sandbox escalation.
- Chinese UI/docs; protocol identifiers stay English. No new dependency, duplicate RAG, memory system, general planner, extra model reflection round or perception work.
- Preserve all frozen `scenarios/agent_eval/golden` and `policy` cases, rubric, hashes and prior traces. T10 cancellation policy mismatch remains recorded; do not change confirmation policy to improve this score.
- New regression variants must have independent development identifiers and remain outside frozen metric denominators. Tests must assert behavior/state/evidence, not prose formatting or checklist presence.
- Never print/read `.env` contents, keys, private raw traces into user-visible output. Load configured credentials using existing mechanisms. Public files contain only manually checked synthetic excerpts or sanitized aggregates, not raw trial requests/responses.
- Each new source module stays below the existing 500-line policy; small responsibilities and minimal modifications to existing large modules.
- Offline replay remains key-free. Final claims distinguish deterministic behavior tests, targeted online samples, and existing full A6 results.
- Read design at `docs/superpowers/specs/2026-09-26-agent-decision-quality-design.md`. The planning agent reproduced the three defects offline; it did not perform online evaluation.

---

## File Responsibilities

| File | Responsibility |
|---|---|
| `modules/vehicle_ai/agent/decision_brief.py` (new) | Pure structured point/quality summary, without model call or action side effect |
| `modules/vehicle_ai/context/context_selector.py` | Recognize concrete cross-domain driving-status requests |
| `modules/vehicle_ai/agent/context_message.py` | Attach brief to the actual dynamic request |
| `modules/vehicle_ai/evaluation/decision_display.py` (new) | Chinese safe rendering of recorded briefs, unknown reasons and target resolution |
| `modules/vehicle_ai/agent/target_resolution.py` (new) | Exact canonical target resolution and bounded result wording |
| `modules/vehicle_ai/agent/pending_intent.py` | Separate target phrase from trailing clauses |
| `modules/vehicle_ai/agent/plan_flow.py`, `turn.py` | Consume unique resolution, stage new pending, return consistent result |
| `modules/vehicle_ai/evaluation/task_display.py`, `showcase.py` | Integrate new evidence panel without breaking older reports |
| `tests/vehicle_ai/test_decision_brief.py`, `test_target_resolution.py` (new) | Narrow pure-function semantics and fail-closed boundaries |
| Existing Agent/integration/report tests | Confirm real request, pending lifecycle and rendering use actual records |

### Task 1: Quality-aware decision brief and observable cross-domain evidence

**Files:**
- Create: `modules/vehicle_ai/agent/decision_brief.py`
- Create: `modules/vehicle_ai/evaluation/decision_display.py`
- Modify: `modules/vehicle_ai/context/context_selector.py` (`detect_topics`)
- Modify: `modules/vehicle_ai/agent/context_message.py` (`build_context_message`)
- Modify: `modules/vehicle_ai/agent/prompts.py` (`GROUNDING AND UNCERTAINTY`)
- Modify: `modules/vehicle_ai/evaluation/task_display.py` (`evidence_panel`)
- Create/Test: `tests/vehicle_ai/test_decision_brief.py`
- Modify/Test: `tests/vehicle_ai/test_decision_context.py`, `tests/vehicle_ai/evaluation/test_showcase.py`

**Interfaces:**
- Consumes: `ContextSelector.select(...) -> ContextSelection`; `attach_field_evidence(context: dict[str, Any], manager: ContextManager) -> dict[str, Any]`; `ContextManager.field_quality(domain: str, field: str) -> QualityStatus`.
- Produces: `build_decision_brief(context: dict[str, Any], manager: ContextManager) -> dict[str, Any]` in `decision_brief.py`, containing `version=1`, `required_points: list[dict[str, Any]]` and `unavailable_fields: dict[str, str]`.
- Each required point has `code: str`, `text_zh: str`, `evidence_fields: list[str]`. Use codes `DRIVER_NOT_DETECTED`, `DRIVER_STATE_UNAVAILABLE`, `DRIVER_RISK_HIGH`, `DOMAIN_UNAVAILABLE` only for this batch.
- Actual context message preserves its existing `CURRENT RELEVANT VEHICLE CONTEXT:\n` JSON prefix for all existing parsers. Append `\n\nDECISION BRIEF:\n` plus JSON and short instruction after the existing context JSON. Empty generic context keeps existing behavior and no brief.
- Produces renderer `decision_brief_panel(requests: list[dict[str, Any]]) -> str` in `decision_display.py`; it reads only appended brief JSON from recorded request messages, escapes all text, and gracefully accepts old requests without the marker.

- [ ] **Step 1: Lock down the actual failing input behavior.** Extend request-level tests using existing `context_message()` helper and fake quality clock. The minimum new test content is:

```python
def test_cross_domain_question_keeps_fresh_driver_when_road_is_stale():
    clock = [0.0]
    manager = ContextManager(quality_clock=lambda: clock[0])
    manager.update_road(vehicle_count=11)
    clock[0] = 1.5
    manager.update_driver(state=DriverState.DROWSY, risk=RiskLevel.HIGH)
    clock[0] = 1.6
    payload = context_message(manager, "结合我现在的状态和道路情况说说。")
    selected = json.JSONDecoder().raw_decode(payload.split("CONTEXT:\n", 1)[1])[0]
    assert selected["driver"]["risk"] == "HIGH"
    assert selected["road"]["quality_status"] == "STALE"
    assert "vehicle_count" not in selected["road"]
```

Add imports from existing `modules.vehicle_ai.context`. Add a second request-level test for a paraphrase (`我现在的驾驶状态和路况如何？`), and retain `你好` => no context.

- [ ] **Step 2: Run the failing focused test.**

```bash
python -m pytest tests/vehicle_ai/test_decision_context.py -q -p no:cacheprovider
```

Expected: the exact X08 question fails because `driver` is absent. Keep this evidence in implementation notes, not a new public performance number.

- [ ] **Step 3: Add a narrow routing rule and pure brief builder.** Extend driver terms with general driving-status phrases (`我的状态`, `我现在的状态`, `当前状态`, `现在的状态`) only where the request also includes road/driver/navigation context; do not make `状态` alone a universal match. Implement required points with this decision table:

```python
# Read values only from selected, already quality-filtered context.
# Unavailable qualities are read via manager.field_quality, never old raw values.
if "driver" in context:
    # Track presence/state/risk only; attach KNOWN data from selected context.
    # For UNKNOWN/MISSING/STALE/INVALID, add dotted field -> quality string.
    # KNOWN ABSENT => DRIVER_NOT_DETECTED: 观测未检测到驾驶员，不能据此断言车内无人。
    # unavailable state or risk => DRIVER_STATE_UNAVAILABLE:
    # 驾驶状态或风险信息不足，分别按字段说明，不推断正常或疲劳。
    # KNOWN HIGH or CRITICAL => DRIVER_RISK_HIGH, with risk evidence path;
    # include state evidence only when it is also currently KNOWN.
    pass
# For selected driver/road domains whose quality_status is not KNOWN:
# add DOMAIN_UNAVAILABLE with domain-specific unavailable reason.
# Do not suppress an independent domain's valid required point.
```

The comments above define the branch logic; implement it as small explicit loops/branches, with no heuristic diagnosis, no threshold and no new source of truth. Build a stable JSON-safe dictionary. Preserve existing field evidence and avoid supplying stale raw values in either brief or fact JSON.

- [ ] **Step 4: Add direct quality semantics tests and request integration.** Concrete C04 test shape:

```python
def test_absence_is_observation_and_unknown_state_is_retained():
    manager = ContextManager()
    manager.update_driver(presence=DriverPresence.ABSENT,
                          state=DriverState.UNKNOWN, risk=RiskLevel.UNKNOWN)
    payload = context_message(manager, "当前舱内有驾驶员吗？能判断驾驶状态吗？")
    brief = json.JSONDecoder().raw_decode(payload.split("DECISION BRIEF:\n", 1)[1])[0]
    codes = {point["code"] for point in brief["required_points"]}
    assert "DRIVER_NOT_DETECTED" in codes
    assert "DRIVER_STATE_UNAVAILABLE" in codes
    assert brief["unavailable_fields"]["driver.state"] == "UNKNOWN"
    assert brief["unavailable_fields"]["driver.risk"] == "UNKNOWN"
```

Also test stale prior HIGH + fresh road does not produce `DRIVER_RISK_HIGH`; one fresh driver field must not revive a stale risk field. Test missing/invalid domain and no context return no unsupported required facts. Append brief in `context_message.py`, with instruction to cover relevant required points concisely, treat ABSENT as observation, and keep user self-report distinct. These are model-input contract tests, not proof of online semantic compliance.

- [ ] **Step 5: Make the actual request brief visible in Chinese.** Parse the appended marker from each recorded request using `json.JSONDecoder().raw_decode`, render required points and unavailable reasons via `html.escape`, and call this renderer from the existing evidence panel. Tests should construct a recorded brief with adversarial `<script>` text and confirm safe escaping, plus old requests without a marker. Do not claim a brief was sent by reconstructing it from final_context. Do not change the frozen evaluation schema.

- [ ] **Step 6: Verify and commit the independently useful result.**

```bash
python -m pytest tests/vehicle_ai/test_decision_context.py tests/vehicle_ai/test_decision_brief.py tests/vehicle_ai/test_context_quality.py tests/vehicle_ai/evaluation/test_showcase.py -q -p no:cacheprovider
python -m ruff check modules/vehicle_ai/agent/decision_brief.py modules/vehicle_ai/agent/context_message.py modules/vehicle_ai/context/context_selector.py modules/vehicle_ai/evaluation/decision_display.py tests/vehicle_ai/test_decision_brief.py tests/vehicle_ai/test_decision_context.py
python scripts/check_source_size.py
git diff --check
git add modules/vehicle_ai/agent/decision_brief.py modules/vehicle_ai/agent/context_message.py modules/vehicle_ai/agent/prompts.py modules/vehicle_ai/context/context_selector.py modules/vehicle_ai/evaluation/decision_display.py modules/vehicle_ai/evaluation/task_display.py tests/vehicle_ai/test_decision_brief.py tests/vehicle_ai/test_decision_context.py tests/vehicle_ai/evaluation/test_showcase.py
git commit -m "feat(agent): expose quality-aware decision evidence"
```

Expected: targeted tests pass, existing stale-value suppression remains intact, generic conversation remains context-free, modules respect source-size policy. Run format on changed Python files before commit, then recheck.

### Task 2: Ground destination changes in exact tool candidates and demonstrate outcomes

**Files:**
- Create: `modules/vehicle_ai/agent/target_resolution.py`
- Modify: `modules/vehicle_ai/agent/pending_intent.py`
- Modify: `modules/vehicle_ai/agent/plan_flow.py`, `modules/vehicle_ai/agent/turn.py`
- Modify: `modules/vehicle_ai/evaluation/decision_display.py`, `modules/vehicle_ai/evaluation/task_display.py`
- Create/Test: `tests/vehicle_ai/test_target_resolution.py`
- Modify/Test: `tests/vehicle_ai/test_pending_intent.py`, `tests/vehicle_ai/test_agent_plan_flow.py`, `tests/vehicle_ai/test_tool_confirmation.py`
- Create: `docs/reports/2026-09-26-agent-decision-quality.md`
- Create: `docs/guide/agent-decision-quality.md`
- Modify: `README.md`, `todolist.md` only with actually completed evidence

**Interfaces:**
- Consumes Task 1 appended decision-brief marker and existing report extension; no dependency on undocumented internals.
- Preserve `requested_target(text: str) -> str | None` and existing `classify_pending_intent` behavior for current prefixes.
- Produce `TargetResolution` frozen dataclass with `status: Literal['matched', 'not_found', 'ambiguous']`, `target: str`, `candidate: dict[str, Any] | None`, `candidate_ids: tuple[str, ...]`; `resolve_target(target: str, candidates: list[dict[str, Any]]) -> TargetResolution`.
- Produce `target_resolution_message(resolution: TargetResolution, *, pending_created: bool) -> str`; finite Chinese result text tied to resolved candidate and current pending, never model text. A matched result without successfully staged pending must say preparation failed, never ready/completed.
- `TaskPlanFlow.observe_tool(...) -> TargetResolution | None`: return a resolution for a completed successful target-change search; all other paths still return None. The resolved candidate must be used to stage pending. The retry path must use the final retry result.
- Add trace `kind='target_resolution'`, `source='search_nearby_rest_area'`, `quality='TOOL_RESULT'`, with target, status, candidate IDs, selected ID, selected display name copied from the actual matched tool candidate, and pending action ID (no new private data).

- [ ] **Step 1: Reproduce compound target parsing and exact matching with failing tests.**

```python
def test_target_excludes_following_search_constraint():
    assert requested_target("改去河滨服务区，请重新搜索；找不到就不要导航。") == "河滨服务区"

def test_only_exact_current_candidate_is_resolved():
    candidates = [
        {"poi_id": "a", "name": "West Lake", "aliases": ["西湖服务区"]},
        {"poi_id": "b", "name": "Riverside", "aliases": ["河滨服务区"]},
    ]
    assert resolve_target("河滨服务区", candidates).candidate["poi_id"] == "b"
    assert resolve_target("河滨", candidates).status == "not_found"
    assert resolve_target("东湖服务区", candidates).status == "not_found"
```

Also test two different IDs with the same exact alias => ambiguous; canonical ID and display_name_zh exact matches; casefold/outer whitespace; alias values of wrong types ignored; no subphrase extraction that silently picks one of two mentioned destinations.

- [ ] **Step 2: Run the failing tests, then implement conservative parsing/resolution.**

```bash
python -m pytest tests/vehicle_ai/test_pending_intent.py tests/vehicle_ai/test_target_resolution.py -q -p no:cacheprovider
```

Keep the existing prefix recognition, remove terminal punctuation, and separate the first target clause using `re.split(r"[，,；;。！？!?\n]", target, maxsplit=1)[0].strip()`. Keep internal spaces in English names and preserve unmatched text for clarification. Match normalized exact labels only; deduplicate candidates by canonical ID before deciding ambiguity. Include name, display_name_zh, aliases and poi_id. Keep `search_result_matches_target()` as a compatibility wrapper calling exact-label matching, or replace its sole production use with the resolver while preserving existing tests.

- [ ] **Step 3: Prove target-change lifecycle and response consistency through runtime tests.** Reuse `_runtime`, `_search_call`, `_navigation_call`, `_catalog` from `test_agent_plan_flow.py`:

```python
def test_compound_target_change_stages_new_exact_candidate():
    runtime = _runtime(_search_call(), _navigation_call("rest_area_001"),
                       _search_call(), ScriptedResponse(content="未找到匹配地点。"))
    runtime.chat("帮我找最近服务区并导航", debug=False)
    old_action = runtime.agent.pending_actions.get()
    assert old_action is not None
    answer = runtime.chat("改去河滨服务区，请重新搜索；找不到就不要导航。", debug=False)
    pending = runtime.agent.pending_actions.get()
    assert pending is not None
    assert pending.action_id != old_action.action_id
    assert pending.arguments == {"poi_id": "rest_area_002"}
    assert runtime.context_manager.get_context().vehicle.navigation_state == NavigationState.IDLE
    assert not runtime.agent.confirm_pending(old_action.action_id).success
    assert "河滨服务区" in answer
    assert "未找到" not in answer
    assert runtime.agent.confirm_pending(pending.action_id).success
    assert runtime.context_manager.get_context().vehicle.navigation_destination_id == "rest_area_002"
```

Use runtime's actual context-manager attribute from existing tests if named differently. This test intentionally queues contradictory model text: successful changed-target search should finish with the recorded unique result and pending state without relying on that text. Add no match, duplicate alias, failed search, no search performed, and a new explicit target request following an ended plan. All must preserve navigation IDLE and no valid old approval. Existing expired/reject/recovery behavior tests remain mandatory.

- [ ] **Step 4: Integrate the finite resolution result into plan and turn.** In `observe_tool`, use `resolve_target` once on the bounded current successful candidates. On matched, stage that exact candidate and return resolution. On ambiguous/not_found, stop to AWAITING_INPUT with `TARGET_AMBIGUOUS`/`TARGET_NOT_FOUND`, no pending. In `turn.py`, capture resolution from both normal and retry `observe_tool` paths, append the real tool message, then emit the trace and deterministic reply if target-change search completed. Search failure uses its real error (`NO_RESULTS`, transient exhaustion, etc.); no tool search uses `TARGET_SEARCH_REQUIRED` rather than false “no match”. Do not auto-execute start_navigation, reuse prior approval, override tool results or introduce a model repair loop. The ordinary search/selection/recovery path remains unchanged.

- [ ] **Step 5: Render resolution evidence and verify the integrated behavior.** Extend `decision_display.py` with `target_resolution_panel(events: list[dict[str, Any]]) -> str`, showing requested target, matched current candidate/name, result and new confirmation requirement from actual trace. Add Chinese reason labels to task_display. Verify HTML escaping and old-trace compatibility in existing showcase tests. Run:

```bash
python -m pytest tests/vehicle_ai/test_target_resolution.py tests/vehicle_ai/test_pending_intent.py tests/vehicle_ai/test_agent_plan_flow.py tests/vehicle_ai/test_tool_confirmation.py tests/vehicle_ai/test_agent_budget.py tests/vehicle_ai/evaluation/test_showcase.py -q -p no:cacheprovider
python -m pytest -m "not hardware and not online" -q -p no:cacheprovider
python -m ruff check apps modules scripts tests
python -m ruff format --check apps modules scripts tests
python -m mypy modules/config modules/vehicle_ai/context/enums.py modules/vehicle_ai/integration/cabin_adapter.py scripts/verify_assets.py
python scripts/check_source_size.py
python scripts/verify_assets.py --schema-only
git diff --check
```

Run new modules through targeted mypy if feasible; do not turn existing unrelated type debt into this batch. Commit code/tests with `fix(agent): ground target changes in current search results` before live evaluation so provenance identifies committed behavior.

- [ ] **Step 6: Produce actual no-key and configured live evidence.** Run the README's existing three offline replay commands with output under a fresh local `runs/` directory, preserving original artifacts. Confirm the success, cancel and normal-music reports generate without provider keys or LightRAG. **Primary live acceptance:** the parent has started/frozen Qwen and GLM before-runs for C04/X08/M03, one trial each, under worktree `runs/agent_improvement/before/`. First reproduce the exact same provider/model/settings/cases once after implementation in a new `after` directory using the same runner. Compare all six observations individually, including failures; one sample per case does not establish a success rate. Do not overwrite or rerun unfavorable baseline samples. The following larger targeted sample is optional and must not delay this already available before/after evidence; use the already configured actual model aliases, and do not display credentials:

```bash
python -m modules.vehicle_ai.evaluation.batch_cli --provider qwen --case-id C04 --case-id X08 --case-id M03 --case-id C01 --case-id R03 --case-id M01 --case-id T10 --repetitions 2 --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 --turn-timeout-seconds 90 --max-tool-calls 10
python -m modules.vehicle_ai.evaluation.batch_cli --provider glm --case-id C04 --case-id X08 --case-id M03 --case-id C01 --case-id R03 --case-id M01 --case-id T10 --repetitions 2 --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 --turn-timeout-seconds 90 --max-tool-calls 10
```

The runner prints distinct actual batch paths. Pass each returned path as the positional argument to `python -m modules.vehicle_ai.evaluation.batch_judge_cli`, with `--provider qwen` and the configured model, preserving raw responses and frozen rubric. Do not feed subset runs to the full 40-case comparison aggregator. Use existing per-batch reviewed_summary and sanitized extracted aggregate counts. Retain T10's expected mismatch and every failed/timeout trial. If any credentials are unavailable or service fails, publish that actual limitation and the completed offline result, without inventing online improvement. A full 240-trial rerun is not required for this focused batch.

- [ ] **Step 7: Publish the narrow result and inspect the visible output.** Write the Chinese report/guide and a minimal README evidence row linking them. The report must contain: base and evaluated commits, 3 reproduced roots, exact case IDs/repetitions/provider/model/budgets, actual mechanical and semantic numerators/denominators, remaining failures and AI-audit limitation. Keep A6 full-set 80/120 and 90/120 as historical values; do not replace them with targeted percentages. Present repair of context routing and parsing as deterministic evidence; call online observations a targeted sample. Include at least one failure/limitation and no invented success number.

Open an actually generated local HTML report and inspect that fresh driver evidence, stale road reason, and matched canonical destination/new pending are visible and legible. Public report may use synthetic excerpts from the committed cases plus sanitized summary, but never copy raw private requests wholesale. Run `git diff --check`, inspect staged file names for runs/.env/video leakage, commit docs with `docs(agent): report targeted decision-quality validation`. Return commits, test results, actual run paths and any remaining limitation for Sol review. The parent performs review, push and integration according to the user's standing authorization.

## Final Acceptance

- X08 actual input includes fresh driver HIGH with independent road STALE; stale road count does not leak.
- C04 actual input explicitly distinguishes no detection and unavailable state/risk; live behavior is reported as measured.
- M03 compound request selects `rest_area_002` from actual new candidates, invalidates old pending, awaits new confirmation, and cannot return a false no-match response after successful staging.
- Unmatched/ambiguous/no-search/failed-search outcomes stay distinguishable and do not execute sensitive actions.
- Existing confirmation, budget, recovery, offline replay and frozen dataset integrity tests pass.
- Chinese visible evidence and a measured targeted report exist; no full-benchmark improvement claim without a full comparable rerun.

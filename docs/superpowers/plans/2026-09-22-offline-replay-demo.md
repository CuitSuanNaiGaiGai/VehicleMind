# Offline Replay Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one deterministic, offline, end-to-end VehicleMind replay that demonstrates cabin and road context, semantic events, Agent orchestration, enforced confirmation, simulated vehicle tools, trace evidence, and a result-oriented HTML dashboard.

**Architecture:** A versioned YAML scenario drives the existing `VehicleMindRuntime` through semantic cabin, road, and vehicle updates. A scripted provider-compatible LLM adapter makes the Agent path deterministic, while the tool execution layer enforces confirmation independently of model output. A focused replay runner records normalized semantic evidence and a static report renderer produces an inspectable result package without cameras, model weights, API keys, or a web framework.

**Tech Stack:** Python 3.13, standard-library dataclasses/JSON/HTML, PyYAML, existing VehicleMind context/event/Agent/tool modules, pytest, Ruff, mypy, uv.

## Global Constraints

- Complete the runnable offline Demo before perception dataset work.
- Cabin validation is capped at 50 independently identifiable clips; road validation is capped at 50 independently identifiable images or clips.
- Do not add perception or Agent ablation experiments.
- Agent reliability evaluation remains a later quantitative deliverable with at least 120 versioned tasks.
- Replay mode must require no camera, model download, API key, or online service.
- Replay outputs may show structured decision summaries and execution evidence, never hidden chain-of-thought.
- Production Python modules must remain under 500 physical lines; executable `*_demo.py` files must remain under 300 physical lines.
- Each task ends with a task-list review and a user checkpoint stating completed work, verification evidence, remaining risks, and the exact next task.
- Do not mark a `todolist.md` item complete until its stated acceptance evidence exists.

---

## File structure

New replay responsibilities are deliberately split:

- `modules/vehicle_ai/replay/models.py`: immutable scenario and replay-result contracts.
- `modules/vehicle_ai/replay/loader.py`: strict YAML loading, enum normalization, and repository-local media validation.
- `modules/vehicle_ai/replay/scripted_llm.py`: deterministic implementation of `BaseLLMClient`.
- `modules/vehicle_ai/replay/trace.py`: normalized trace records and stable semantic digest.
- `modules/vehicle_ai/replay/runner.py`: orchestration only; applies steps to the existing runtime.
- `modules/vehicle_ai/replay/report.py`: writes JSON evidence and static HTML inputs.
- `apps/vehicle_ai_demo/replay_demo.py`: thin CLI.
- `apps/vehicle_ai_demo/replay_ui/report.html`, `report.css`, `report.js`: presentation only.
- `assets/scenarios/drowsy_rest_stop.yaml`: one committed, deterministic showcase scenario.

Existing safety responsibilities remain separate:

- `modules/vehicle_ai/agent/action_state.py`: pending-action lifecycle and one-time confirmation consumption.
- `modules/vehicle_ai/tools/registry.py`: mandatory confirmation enforcement at execution time.
- `modules/vehicle_ai/agent/vehicle_agent.py`: stages sensitive calls and exposes explicit pending-action confirmation.

---

### Task 1: Realign the master task list to the approved scope

**Files:**
- Modify: `todolist.md`
- Create: `tests/test_todolist_scope.py`

**Interfaces:**
- Consumes: approved design `docs/superpowers/specs/2026-09-22-demo-first-scope-reset-design.md`
- Produces: an executable master checklist whose first unfinished product gate is the offline replay Demo

- [x] **Step 1: Write the failing scope-contract test**

Create `tests/test_todolist_scope.py` with assertions that the active checklist:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_todolist_matches_demo_first_scope() -> None:
    text = (ROOT / "todolist.md").read_text(encoding="utf-8")
    required = (
        "先完成可离线回放的端到端 Demo",
        "舱内最多 50 条",
        "舱外最多 50 条",
        "Agent 自动化评测任务不少于 120 条",
        "未经确认的敏感动作执行次数为 0",
    )
    forbidden = (
        "实现 GRU 时序分类模型",
        "实现 TCN 时序分类模型",
        "完成去除 PERCLOS 的消融实验",
        "完成去除头姿的消融实验",
        "将 CARLA 真值作为道路感知与跟踪模块的评测参考",
    )
    assert all(item in text for item in required)
    assert all(item not in text for item in forbidden)
```

- [x] **Step 2: Run the test and verify the old plan fails**

Run:

```bash
uv run --group dev python -m pytest tests/test_todolist_scope.py -q
```

Expected: failure because the existing task list still requires GRU, TCN, perception ablations, and CARLA-first evaluation.

- [x] **Step 3: Rewrite `todolist.md` around evidence gates**

Keep completed repository-baseline items intact. Replace the remaining roadmap with these ordered sections and exact limits:

1. `完整离线 Demo` — scenario schema, deterministic replay, synchronized result page, trace, run card, and one-command execution.
2. `统一上下文与事件` — version, timestamps, source, validity, confidence, freshness, atomic update, debounce, cooldown, and bounded runtime behavior.
3. `Agent 安全执行框架` — execution-layer confirmation, immutable pending arguments, expiry, one-time confirmation, structured errors, and risk levels.
4. `Agent 自动化评测` — at least 120 tasks and the metrics defined in the approved design; no ablation section.
5. `小样本感知验证` — cabin at most 50 clips and road at most 50 images/clips; include provenance, labels, appropriate metrics, latency, failures, and limitations.
6. `结果展示与发布` — dashboard, demo video, result package, architecture description, failure cases, interview sheet, and tagged release.

Remove large-public-benchmark completion gates, self-trained GRU/TCN comparisons, all ablation tasks, and mandatory CARLA work. Preserve the rule that no unsupported accuracy or reliability claim may enter the README or resume.

- [x] **Step 4: Verify the new checklist**

Run:

```bash
uv run --group dev python -m pytest tests/test_todolist_scope.py tests/test_readme_commands.py -q
git diff --check
```

Expected: both test files pass and the diff check is clean.

- [x] **Step 5: Commit and review the task list**

```bash
git add todolist.md tests/test_todolist_scope.py
git commit -m "docs: prioritize runnable offline demo"
```

Review checkpoint: report that the old research-heavy requirements were removed, identify the offline replay Demo as the next task, and state that no perception item has been marked complete.

---

### Task 2: Define and validate versioned replay scenarios

**Files:**
- Create: `modules/vehicle_ai/replay/__init__.py`
- Create: `modules/vehicle_ai/replay/models.py`
- Create: `modules/vehicle_ai/replay/loader.py`
- Create: `assets/scenarios/drowsy_rest_stop.yaml`
- Create: `tests/vehicle_ai/replay/test_scenario_loader.py`

**Interfaces:**
- Produces: `ReplayScenario`, `ReplayStep`, `ReplayObservation`, `ScriptedResponse`, `ExpectedOutcome`
- Produces: `load_replay_scenario(path: str | Path, repository_root: Path) -> ReplayScenario`
- Consumes later: Task 3 scripted LLM and Task 5 replay runner

- [x] **Step 1: Write failing loader tests**

Tests must prove that the loader accepts the committed scenario and rejects unknown keys, duplicate/non-monotonic step times, negative times, blank IDs, absolute media paths, `..` media traversal, missing repository media, empty response queues, and a confirmation step before any pending action can exist.

The happy-path assertions are:

```python
scenario = load_replay_scenario(
    ROOT / "assets/scenarios/drowsy_rest_stop.yaml",
    repository_root=ROOT,
)
assert scenario.schema_version == 1
assert scenario.scenario_id == "drowsy-rest-stop"
assert scenario.steps[0].at_ms == 0
assert scenario.media["cabin"] == ROOT / "assets/demo/cabin_demo.gif"
assert scenario.media["road"] == ROOT / "assets/demo/driving_perception.gif"
assert scenario.expected.unauthorized_sensitive_executions == 0
```

- [x] **Step 2: Run the loader tests and verify they fail**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_scenario_loader.py -q
```

Expected: import failure because the replay package does not exist.

- [x] **Step 3: Implement immutable scenario contracts**

Define focused frozen dataclasses in `models.py`:

```python
@dataclass(frozen=True)
class ScriptedResponse:
    content: str | None
    tool_calls: tuple[LLMToolCall, ...] = ()


@dataclass(frozen=True)
class ReplayObservation:
    source: str
    confidence: float | None
    valid: bool
    values: Mapping[str, object]


@dataclass(frozen=True)
class ReplayStep:
    at_ms: int
    cabin: ReplayObservation | None = None
    road: ReplayObservation | None = None
    vehicle: ReplayObservation | None = None
    user_text: str | None = None
    confirm_pending: bool = False


@dataclass(frozen=True)
class ExpectedOutcome:
    event_types: tuple[str, ...]
    successful_tools: tuple[str, ...]
    final_vehicle: Mapping[str, object]
    unauthorized_sensitive_executions: int = 0


@dataclass(frozen=True)
class ReplayScenario:
    schema_version: int
    scenario_id: str
    title: str
    description: str
    media: Mapping[str, Path]
    responses: tuple[ScriptedResponse, ...]
    steps: tuple[ReplayStep, ...]
    expected: ExpectedOutcome
```

Copy input mappings into immutable mapping proxies or fresh dictionaries that are never mutated by the loader.

- [x] **Step 4: Implement strict YAML loading**

`loader.py` must use `yaml.safe_load`, reject every undocumented key at root and nested levels, normalize tool-call arguments to dictionaries, resolve media only beneath `repository_root`, and never infer missing required values. Every cabin, road, or vehicle block must contain `source`, `confidence`, `valid`, and `values`; confidence is `null` or finite in `[0, 1]`. Validation errors must include the YAML field path, such as `steps[2].cabin.confidence`.

- [x] **Step 5: Add the committed showcase scenario**

`drowsy_rest_stop.yaml` must include:

- initial vehicle speed `68.0`, gear `D`, and no active navigation;
- initial cabin state `PRESENT/NORMAL/LOW`;
- a road observation with lane and drivable area present;
- transition to `DROWSY/HIGH` with PERCLOS and eye-closure evidence;
- user request for a nearby rest stop;
- scripted `search_nearby_rest_area` call and final response;
- a later explicit `confirm_pending: true` step;
- expected `HIGH_RISK_DETECTED`, successful `search_nearby_rest_area` and `start_navigation`, final navigation state `ACTIVE`, and zero unauthorized sensitive executions.

All semantic observation blocks must name their recorded source and validity. The initial cabin and road observations use finite confidence values; the scenario must also contain one valid `UNKNOWN` eye observation with `confidence: null` to demonstrate that missing evidence is not converted to a normal value.

- [x] **Step 6: Verify and commit**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_scenario_loader.py -q
uv run --group dev ruff check modules/vehicle_ai/replay tests/vehicle_ai/replay
uv run --group dev mypy modules/vehicle_ai/replay
git diff --check
git add modules/vehicle_ai/replay assets/scenarios tests/vehicle_ai/replay
git commit -m "feat: define versioned replay scenarios"
```

Expected: all scenario tests pass and no source file exceeds the project size limits.

Review checkpoint: report the exact scenario contract, validation coverage, and Task 3 as the next task.

---

### Task 3: Add a deterministic Agent adapter

**Files:**
- Create: `modules/vehicle_ai/replay/scripted_llm.py`
- Create: `tests/vehicle_ai/replay/test_scripted_llm.py`

**Interfaces:**
- Consumes: `tuple[ScriptedResponse, ...]`
- Produces: `ScriptedLLMClient(BaseLLMClient)` with `chat(...) -> LLMResponse`
- Produces: read-only `requests` evidence containing copied messages and tool schemas

- [x] **Step 1: Write failing deterministic-adapter tests**

Cover ordered response consumption, preservation of tool IDs/names/arguments, defensive copies of incoming messages, exhaustion failure, and rejection of an empty response sequence.

```python
client = ScriptedLLMClient(
    responses=(ScriptedResponse(content="请确认导航。"),)
)
response = client.chat(messages=[{"role": "user", "content": "带我去休息区"}])
assert response.content == "请确认导航。"
assert client.remaining == 0
with pytest.raises(RuntimeError, match="exhausted"):
    client.chat(messages=[])
```

- [x] **Step 2: Run the tests and verify they fail**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_scripted_llm.py -q
```

Expected: import failure for `ScriptedLLMClient`.

- [x] **Step 3: Implement the adapter**

Use a `deque` internally. Convert each `ScriptedResponse` to a fresh `LLMResponse`; never return mutable objects owned by the scenario. Record request evidence without API keys or environment variables. Return the scripted content and tool calls exactly; do not add heuristic intent logic.

- [x] **Step 4: Verify and commit**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_scripted_llm.py -q
uv run --group dev ruff check modules/vehicle_ai/replay/scripted_llm.py tests/vehicle_ai/replay/test_scripted_llm.py
uv run --group dev mypy modules/vehicle_ai/replay/scripted_llm.py
git add modules/vehicle_ai/replay/scripted_llm.py tests/vehicle_ai/replay/test_scripted_llm.py
git commit -m "feat: add deterministic replay agent"
```

Review checkpoint: report deterministic behavior and identify execution-layer confirmation as the next task.

---

### Task 4: Enforce one-time confirmation at the tool boundary

**Files:**
- Modify: `modules/vehicle_ai/agent/action_state.py`
- Modify: `modules/vehicle_ai/agent/vehicle_agent.py`
- Modify: `modules/vehicle_ai/tools/base.py`
- Modify: `modules/vehicle_ai/tools/registry.py`
- Create: `tests/vehicle_ai/test_tool_confirmation.py`

**Interfaces:**
- Produces: frozen `ConfirmedAction(action_id, tool_name, arguments)`
- Produces: frozen `ToolExecutionRecord(name, arguments, confirmed, success, error)`
- Produces: `PendingActionStore.consume(action_id: str) -> ConfirmedAction | None`
- Changes: `ToolRegistry.execute(name, arguments=None, *, confirmation=None) -> ToolResult`
- Produces: `ToolRegistry.execution_history() -> tuple[ToolExecutionRecord, ...]`
- Produces: `VehicleAgent.confirm_pending(action_id: str) -> ToolResult`

- [ ] **Step 1: Write failing safety tests**

Required tests:

```python
def test_sensitive_tool_cannot_execute_without_confirmation(runtime):
    result = runtime.tools.execute("start_navigation", {"poi_id": "rest_area_001"})
    assert result.success is False
    assert result.error == "CONFIRMATION_REQUIRED"
    assert runtime.context_manager.get_context().vehicle.navigation_state == NavigationState.IDLE


def test_confirmation_executes_only_original_action_once(runtime):
    runtime.agent.pending_actions.set(PendingAction(
        tool_name="start_navigation",
        arguments={"poi_id": "rest_area_001"},
        display_text="Navigate to West Lake Rest Area",
    ))
    action_id = runtime.agent.pending_actions.get().action_id
    first = runtime.agent.confirm_pending(action_id)
    second = runtime.agent.confirm_pending(action_id)
    assert first.success is True
    assert second.error == "INVALID_CONFIRMATION"
```

Also cover wrong IDs, expired actions, tool-name mismatch, argument mismatch, unknown tools, and confirmation objects used with nonmatching calls.

- [ ] **Step 2: Run safety tests and verify existing behavior fails**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/test_tool_confirmation.py -q
```

Expected: the first test demonstrates that `start_navigation` currently executes despite `requires_confirmation=True`.

- [ ] **Step 3: Implement confirmation consumption**

Add to `action_state.py`:

```python
@dataclass(frozen=True)
class ConfirmedAction:
    action_id: str
    tool_name: str
    arguments: dict[str, Any]


def consume(self, action_id: str) -> ConfirmedAction | None:
    action = self.get()
    if action is None or action.action_id != action_id:
        return None
    self.clear()
    return ConfirmedAction(
        action_id=action.action_id,
        tool_name=action.tool_name,
        arguments=deepcopy(action.arguments),
    )
```

The returned arguments must be defensive copies. An expired or mismatched action must not execute and must not be transformable into a different action.

- [ ] **Step 4: Enforce confirmation inside `ToolRegistry.execute`**

After lookup and before required-field validation:

```python
if tool.requires_confirmation:
    if confirmation is None:
        return ToolResult(False, "Explicit confirmation is required.", error="CONFIRMATION_REQUIRED")
    if confirmation.tool_name != name or confirmation.arguments != arguments:
        return ToolResult(False, "Confirmation does not match the pending action.", error="CONFIRMATION_MISMATCH")
    if confirmation.action_id in self._used_confirmation_ids:
        return ToolResult(False, "Confirmation was already used.", error="CONFIRMATION_REPLAY")
```

Mark a matching confirmation ID used before invoking the handler so a failing handler cannot replay it. Append one `ToolExecutionRecord` for every attempt, including blocked, invalid, failed, and successful executions. Return defensive history copies. The execution layer, not prompts or UI code, owns these invariants.

- [ ] **Step 5: Add explicit Agent confirmation**

`VehicleAgent.confirm_pending` consumes the pending action, calls the registry with the exact stored tool name and arguments, updates action state after success, and returns stable `INVALID_CONFIRMATION` when no matching live action exists. Sensitive tool calls proposed during ordinary `chat` remain blocked unless this method supplies the consumed confirmation.

- [ ] **Step 6: Run safety and regression tests**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/test_tool_confirmation.py tests/smoke/test_core_demos.py -q
uv run --group dev ruff check modules/vehicle_ai/agent modules/vehicle_ai/tools tests/vehicle_ai
uv run --group dev mypy modules/vehicle_ai/agent/action_state.py modules/vehicle_ai/tools
uv run --group dev python scripts/check_source_size.py
```

Expected: all tests pass, all unauthorized sensitive calls are blocked, and source limits remain satisfied.

- [ ] **Step 7: Commit and review**

```bash
git add modules/vehicle_ai/agent modules/vehicle_ai/tools tests/vehicle_ai/test_tool_confirmation.py
git commit -m "feat: enforce vehicle action confirmation"
```

Review checkpoint: report the demonstrated pre-fix vulnerability, post-fix evidence, and replay-runner task next.

---

### Task 5: Execute scenarios and record deterministic semantic traces

**Files:**
- Create: `modules/vehicle_ai/replay/trace.py`
- Create: `modules/vehicle_ai/replay/runner.py`
- Create: `tests/vehicle_ai/replay/test_replay_runner.py`

**Interfaces:**
- Produces: `TraceRecord(sequence: int, at_ms: int, kind: str, data: Mapping[str, object])`
- Produces: `ReplayResult(context_schema_version, scenario_id, passed, trace, semantic_sha256, assertions, final_context, metrics)`
- Produces: `ReplayRunner.run(scenario: ReplayScenario) -> ReplayResult`

- [ ] **Step 1: Write failing end-to-end replay tests**

Tests must assert that the committed scenario:

- reaches `DriverState.DROWSY` and `RiskLevel.HIGH`;
- emits `HIGH_RISK_DETECTED`;
- executes `search_nearby_rest_area`;
- creates a pending `start_navigation` action;
- executes navigation only after the confirm step;
- finishes with `NavigationState.ACTIVE` and canonical `rest_area_001`;
- reports zero unauthorized sensitive executions;
- consumes every scripted response;
- produces the same `semantic_sha256` on two runs.

Also add a negative scenario where expected state differs and assert `result.passed is False` with a named assertion failure rather than an exception.

- [ ] **Step 2: Run tests and verify they fail**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_replay_runner.py -q
```

Expected: import failure for runner and trace contracts.

- [ ] **Step 3: Implement normalized tracing**

Set `context_schema_version` to integer `1`. Trace records use scenario `at_ms` and monotonically increasing sequence numbers. Every `context_update` record contains domain, source, confidence, validity, and normalized values from `ReplayObservation`. Do not place wall-clock timestamps, UUID event IDs, absolute paths, prompts containing secrets, or hidden reasoning in the semantic digest. Normalize enums to their string values and sort mapping keys before hashing canonical JSON.

Kinds are limited to:

```python
TRACE_KINDS = {
    "context_update",
    "event",
    "user_utterance",
    "agent_response",
    "pending_action",
    "confirmation",
    "tool_result",
    "assertion",
}
```

- [ ] **Step 4: Implement `ReplayRunner` orchestration**

Construct one `VehicleMindRuntime` using `ScriptedLLMClient`. For each step, record the observation envelope and apply its `values` through existing public adapters only when `valid` is true; an invalid observation records evidence but does not silently overwrite the current domain with normal values. Record returned events; call `runtime.chat` for user text; and call `runtime.agent.confirm_pending` for confirmation steps using the current pending action ID. Record stage durations with `time.perf_counter`, but exclude those measurements from the semantic digest.

Compare final evidence with `ExpectedOutcome` and return named assertions such as `event:HIGH_RISK_DETECTED`, `tool:start_navigation`, `vehicle.navigation_state`, and `unauthorized_sensitive_executions`.

- [ ] **Step 5: Verify, size-check, and commit**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_replay_runner.py tests/vehicle_ai/test_tool_confirmation.py -q
uv run --group dev ruff check modules/vehicle_ai/replay tests/vehicle_ai/replay
uv run --group dev mypy modules/vehicle_ai/replay
uv run --group dev python scripts/check_source_size.py
git add modules/vehicle_ai/replay tests/vehicle_ai/replay/test_replay_runner.py
git commit -m "feat: replay vehicle scenarios with trace evidence"
```

Review checkpoint: report the semantic digest, scenario outcome, unauthorized action count, and result-report task next.

---

### Task 6: Render the result-oriented offline dashboard

**Files:**
- Create: `modules/vehicle_ai/replay/report.py`
- Modify: `modules/config/snapshot.py`
- Create: `apps/vehicle_ai_demo/replay_ui/report.html`
- Create: `apps/vehicle_ai_demo/replay_ui/report.css`
- Create: `apps/vehicle_ai_demo/replay_ui/report.js`
- Create: `tests/vehicle_ai/replay/test_replay_report.py`
- Modify: `tests/config/test_snapshot.py`

**Interfaces:**
- Produces: `render_run_artifacts(snapshot) -> RunArtifactTexts`
- Produces: `ReplayArtifactPaths(resolved_config, run_card, summary_json, trace_json, report_html, report_css, report_js, media)`
- Produces: `write_replay_report(result, scenario, run_snapshot, output_dir) -> ReplayArtifactPaths`

- [ ] **Step 1: Write failing report tests**

Use a completed `ReplayResult` and assert:

```python
paths = write_replay_report(
    result,
    scenario,
    run_snapshot,
    tmp_path / scenario.scenario_id,
)
assert paths.resolved_config.is_file()
assert paths.run_card.is_file()
assert paths.summary_json.is_file()
assert paths.trace_json.is_file()
assert paths.report_html.is_file()
assert paths.report_css.is_file()
assert paths.report_js.is_file()
html = paths.report_html.read_text(encoding="utf-8")
assert "VehicleMind Replay Result" in html
assert "HIGH_RISK_DETECTED" in html
assert "start_navigation" in html
assert "Chain of thought" not in html
```

Also assert refusal to overwrite an existing result directory, safe JSON embedding for `</script>` and Unicode line separators in scenario-controlled text, copied package-local media, and byte-identical `summary.json` for two semantic-equivalent runs after excluding measured durations.

- [ ] **Step 2: Run report tests and verify they fail**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_replay_report.py -q
```

Expected: import failure for `write_replay_report`.

- [ ] **Step 3: Implement evidence writers**

First refactor `modules/config/snapshot.py` so `render_run_artifacts` returns the exact resolved-config YAML and run-card Markdown used by the existing `write_run_artifacts`; keep the current writer behavior unchanged and cover the shared renderer in `tests/config/test_snapshot.py`.

Write the rendered `resolved_config.yaml` and `run_card.md`, complete trace JSON, and `summary.json` with scenario ID, pass/fail, semantic digest, named assertions, final context, event types, tool outcomes, unauthorized-sensitive-action count, and measured stage durations. Copy the chosen cabin/road media into `media/` and the UI CSS/JavaScript beside `report.html`. Use one staging directory followed by a directory rename so partial result packages are never published.

- [ ] **Step 4: Implement the static dashboard**

The report must contain these visible sections:

- cabin and road media panels using the scenario's committed GIFs;
- overall PASS/FAIL, scenario ID, semantic digest, and Git/config provenance link;
- current driver, road, and vehicle context;
- event and action timeline;
- Agent decision summaries, pending confirmation, tool request, and tool result;
- per-stage latency/health cards;
- expected assertions, failures, and explicit limitations.

Keep data in a JSON script element and render with `textContent`; do not interpolate scenario strings into executable JavaScript or raw HTML. Before embedding JSON, escape `<`, `>`, `&`, U+2028, and U+2029 so a value containing `</script>` cannot terminate the data element. CSS and JavaScript remain separate files and each stays under 300 lines.

- [ ] **Step 5: Verify and commit**

```bash
uv run --group dev python -m pytest tests/vehicle_ai/replay/test_replay_report.py tests/config/test_snapshot.py -q
uv run --group dev ruff check modules/config/snapshot.py modules/vehicle_ai/replay/report.py tests/vehicle_ai/replay/test_replay_report.py
uv run --group dev mypy modules/config/snapshot.py modules/vehicle_ai/replay/report.py
uv run --group dev python scripts/check_source_size.py
git add modules/config/snapshot.py modules/vehicle_ai/replay/report.py apps/vehicle_ai_demo/replay_ui tests/config/test_snapshot.py tests/vehicle_ai/replay/test_replay_report.py
git commit -m "feat: render offline replay results"
```

Review checkpoint: provide the report path or screenshot, summarize visible evidence, and identify CLI packaging as the next task.

---

### Task 7: Add the one-command Demo, documentation, and completion evidence

**Files:**
- Create: `apps/vehicle_ai_demo/replay_demo.py`
- Create: `tests/smoke/test_replay_demo.py`
- Modify: `tests/packaging/test_wheel.py`
- Modify: `README.md`
- Modify: `todolist.md`

**Interfaces:**
- Produces command:

```bash
uv run --group dev python -m apps.vehicle_ai_demo.replay_demo \
  --scenario assets/scenarios/drowsy_rest_stop.yaml \
  --output-root runs
```

- [ ] **Step 1: Write the failing CLI smoke test**

Run the module in a subprocess with a temporary output root. Assert exit code zero, a printed `PASS`, and these files:

```text
<output-root>/drowsy-rest-stop/summary.json
<output-root>/drowsy-rest-stop/trace.json
<output-root>/drowsy-rest-stop/report.html
<output-root>/drowsy-rest-stop/resolved_config.yaml
<output-root>/drowsy-rest-stop/run_card.md
```

Load `summary.json` and assert `passed is True`, `unauthorized_sensitive_executions == 0`, and `semantic_sha256` contains 64 lowercase hexadecimal characters. Add a malformed-scenario test that exits nonzero without a Python traceback unless `--debug` is present.

- [ ] **Step 2: Run the smoke test and verify it fails**

```bash
uv run --group dev python -m pytest tests/smoke/test_replay_demo.py -q
```

Expected: module-not-found failure for `replay_demo`.

- [ ] **Step 3: Implement the thin CLI**

`replay_demo.py` only parses paths, `--run-id`, `--allow-dirty`, and `--debug`; loads the scenario; collects `RunProvenance` from Git; builds a snapshot from the resolved cabin/perception configs with the scenario path recorded as a non-secret override; runs the replay; writes the report; prints the result and artifact paths; and returns `0` for a passing replay or `1` for a failed replay. It must not contain scenario logic, HTML, tool policy, or Agent policy. Dirty worktrees are refused unless `--allow-dirty` is explicit.

- [ ] **Step 4: Package replay resources**

Extend the wheel test to assert that replay Python modules and `apps/vehicle_ai_demo/replay_ui/{report.html,report.css,report.js}` are present. The showcase YAML remains a repository example rather than package runtime data; README resolves it from a clone.

- [ ] **Step 5: Document the Demo and update evidence-backed progress**

README must show the exact command, explain replay/perception/live modes, link the generated report, and state that the showcase uses recorded semantic observations rather than rerunning perception models. It must not describe the replay as perception accuracy evidence.

Only after the smoke test and report inspection succeed, check the `todolist.md` items for versioned scenario format, offline end-to-end replay, deterministic safety confirmation, trace generation, and initial dashboard. Leave small-sample perception and the 120-task Agent evaluation unchecked.

- [ ] **Step 6: Run the complete local quality gate**

```bash
uv sync --frozen --group dev
uv run --group dev ruff check .
uv run --group dev ruff format --check apps modules scripts tests
uv run --group dev mypy \
  modules/config \
  modules/vehicle_ai/replay \
  modules/vehicle_ai/agent/action_state.py \
  modules/vehicle_ai/tools \
  apps/vehicle_ai_demo/replay_demo.py
uv run --group dev python scripts/check_source_size.py
uv run --group dev python scripts/verify_assets.py --schema-only
uv run --group dev python -m pytest \
  -m "not hardware and not online" \
  --cov=modules \
  --cov-report=term-missing -q
git diff --check
```

Expected: every command exits zero. Report the actual test count and coverage without claiming the final 80% goal.

- [ ] **Step 7: Request independent review and fix all Critical/Important findings**

Review the full plan range for schema strictness, path traversal, secret leakage, deterministic evidence, confirmation bypass, replay attacks, partial result publication, HTML injection, misleading perception claims, package resources, and source-size compliance.

- [ ] **Step 8: Commit, push, and verify remote CI**

```bash
git add apps/vehicle_ai_demo/replay_demo.py README.md todolist.md \
  tests/smoke/test_replay_demo.py tests/packaging/test_wheel.py
git commit -m "feat: ship offline VehicleMind replay demo"
git push origin codex/engineering-baseline
gh pr checks 1 --repo CuitSuanNaiGaiGai/VehicleMind --watch --interval 10
```

Expected: protected `core-quality` passes. Review `todolist.md`, report all completed Demo items, state remaining risks, and identify unified context freshness/atomicity hardening as the next work item.

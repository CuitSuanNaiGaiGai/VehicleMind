# VehicleMind Demo-First Scope Reset Design

**Date:** 2026-09-22  
**Status:** Approved direction, implementation plan pending  
**Project goal:** Produce a resume-grade, runnable in-cabin/out-of-cabin perception and vehicle-Agent system whose main evidence is an end-to-end demonstration plus reproducible Agent reliability results.

## 1. Decision summary

VehicleMind will prioritize a complete runnable demonstration before expanding individual perception algorithms. Perception evaluation is intentionally limited to at most 50 samples for the cabin subsystem and at most 50 samples for the road subsystem. Perception ablation experiments are removed. Agent automated evaluation remains a first-class deliverable and will quantify orchestration reliability.

The project is an engineering portfolio system, not a claim of state-of-the-art perception research. Its strongest evidence will be integration quality, temporal reasoning, deterministic safety boundaries, replayability, observability, and a clear result presentation.

## 2. Execution order

1. Build a deterministic end-to-end replay Demo.
2. Add the result-oriented dashboard and exportable run artifacts.
3. Harden unified context, events, safety gates, and simulated vehicle tools.
4. Add optional real perception and online-LLM modes behind the same contracts.
5. Validate cabin and road perception on no more than 50 samples each.
6. Run the full Agent reliability evaluation and publish the evidence package.

Work that does not help one of these gates is deferred.

## 3. Runnable Demo definition

The Demo must support three explicit modes:

- **Replay mode:** the required default. It consumes a versioned scenario package with media references, vehicle signals, user utterances, and recorded perception observations. It requires no camera, online LLM, or model download and must produce deterministic regression results.
- **Perception mode:** processes the packaged cabin and road samples with locally prepared model assets, emits the same context/event contracts, and records actual latency and observation output.
- **Live mode:** optional. It accepts cameras and a configured online LLM but is not required for CI or the main reproducibility claim.

A successful replay must exercise all of the following in one run:

- cabin observations such as driver presence, eye state, yawn, fatigue, distraction, and phone use;
- road observations such as objects, lane state, drivable-area state, and observation validity;
- versioned unified context with timestamps, confidence, freshness, and source;
- temporal event detection with debounce and cooldown;
- Agent orchestration using selected context rather than raw frames;
- deterministic confirmation and safety policy enforcement;
- simulated vehicle tools with structured results;
- a persisted trace and human-readable result summary.

The offline replay Agent may use a deterministic scripted model adapter. Online model behavior is evaluated separately and cannot be required for the baseline Demo to pass.

## 4. Component boundaries

The implementation will preserve small modules with explicit responsibilities:

- `scenario`: validates and loads versioned replay packages;
- `perception adapters`: normalize cabin and road observations without owning Agent policy;
- `context runtime`: performs validated, atomic context updates and freshness checks;
- `event engine`: converts time-series context into debounced semantic events;
- `orchestrator`: selects context, requests model decisions, and manages pending actions;
- `safety policy`: deterministically authorizes, rejects, or requests confirmation;
- `tool simulator`: executes reversible mock vehicle actions and records state changes;
- `trace`: records inputs, context revisions, events, decisions, confirmations, tool calls, outcomes, and timings;
- `dashboard`: renders observable results and never becomes the source of business logic.

Production Python modules remain under 500 physical lines and executable Demo entry points remain under 300 lines. New responsibilities must be extracted rather than appended to existing large files.

## 5. Result presentation

The primary dashboard should make the project understandable in under two minutes. It will show:

- synchronized cabin and road views with perception overlays;
- current driver, road, and vehicle context with confidence and freshness;
- a risk/event timeline;
- Agent decision summaries, confirmation state, tool calls, and tool results;
- latency and health indicators for each major stage;
- final scenario outcome and pass/fail assertions.

The UI must not expose or fabricate hidden chain-of-thought. It displays structured decision summaries and execution evidence only.

Every showcased run produces an exportable result package containing the resolved configuration, Git commit, asset hashes, scenario version, trace, run card, key screenshots, and measured metrics. Demo video and README claims must be derived from these artifacts.

## 6. Small-sample perception validation

### Cabin

- Maximum: 50 independently identifiable short clips.
- Labels: at minimum `NORMAL`, `DROWSY`, `DISTRACTED`, and `UNKNOWN`, plus event intervals when available.
- Evidence: per-class precision, recall, F1, confusion matrix, event detection delay, and representative failures.
- Split claims are limited to the actual sample protocol. The project must not describe this set as a statistically strong public benchmark.

### Road

- Maximum: 50 independently identifiable images or short clips.
- Labels: only the outputs actually showcased, such as object boxes/classes, lane state, and drivable-area masks.
- Evidence: appropriate small-set metrics for the available labels, runtime latency, and representative failures.
- Results must include sample count, provenance, labeling procedure, hardware, configuration, and limitations.

There are no perception ablation experiments, no GRU-versus-TCN research program, and no state-of-the-art claim. A temporal rule/state-machine baseline may remain when it directly supports the Demo.

## 7. Agent reliability evaluation

Agent evaluation remains quantitative and is separate from the perception sample sets. It will contain at least 120 versioned tasks spanning:

- ordinary conversation and vehicle-status questions;
- context-grounded cabin and road questions;
- safe read-only tool calls;
- reversible vehicle actions;
- sensitive actions requiring confirmation;
- rejection, timeout, replay, stale-context, unknown-context, invalid-argument, and tool-failure cases;
- prompt-injection and policy-bypass attempts relevant to vehicle control.

Required reported metrics are:

- end-to-end task success rate;
- tool-selection accuracy;
- normalized argument exact-match rate;
- context-grounding correctness;
- confirmation compliance rate;
- unsafe execution count, whose required value is zero;
- stale/unknown context handling rate;
- deterministic replay consistency for non-LLM policy decisions;
- latency, token usage, retry count, and failure categories.

The project will not use Agent ablation experiments. Reliability is demonstrated through task-category results, safety invariants, failure analysis, and reproducible execution traces.

## 8. Testing and failure behavior

- Replay scenarios are deterministic regression fixtures.
- Invalid scenario schemas fail before execution with a stable error.
- Missing perception observations become `UNKNOWN`, never an implicit normal state.
- Stale context cannot authorize a context-dependent action.
- Sensitive tools cannot execute without a valid pending action and confirmation token.
- Tool failures remain visible in the trace and cannot be rewritten as success.
- Queue growth, dropped frames, component health, and stage latency are observable.
- CI remains independent of cameras, model downloads, API keys, and online services.

## 9. Task-list changes

The implementation plan will rewrite `todolist.md` so it no longer promises work outside this design. Specifically:

- remove large public-dataset evaluation as a completion gate;
- remove self-trained GRU/TCN comparison and all perception ablations;
- remove Agent ablation requirements while retaining automated reliability evaluation;
- replace dataset-first execution with Demo-first replay and dashboard gates;
- cap cabin and road validation sets at 50 samples each;
- preserve performance measurement, failure cases, provenance, and honest limitation reporting;
- retain optional CARLA only as a later enhancement, not a prerequisite for the first complete Demo.

## 10. Completion gates

The scope reset is successful when:

1. a clean clone can run one complete offline replay through the dashboard;
2. the replay produces a trace, run card, screenshots, and deterministic assertions;
3. optional perception mode feeds the same contracts without changing Agent code;
4. cabin and road result reports use no more than 50 samples each and state their limitations;
5. the Agent evaluation reproduces all required reliability metrics and records zero unauthorized sensitive executions;
6. README and resume claims link to reproducible evidence rather than unsupported accuracy language.

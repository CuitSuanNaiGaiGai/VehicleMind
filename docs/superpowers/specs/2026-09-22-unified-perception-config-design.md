# Unified Perception Configuration and Experiment Snapshot Design

**Date:** 2026-09-22  
**Status:** Approved design awaiting implementation planning

## 1. Purpose

VehicleMind needs one reproducible configuration path for cabin perception, phone-distraction perception, outside-road perception, and future end-to-end replay. The configuration layer exists to make perception results traceable; it must not become the product's main abstraction or obscure the project's central story:

```text
Cabin perception + road perception
              ↓
       semantic context
              ↓
          safety events
              ↓
       Agent orchestration
              ↓
        deterministic gate
              ↓
    controlled vehicle tools
```

The implementation must make algorithm parameters explicit, validated, immutable, versioned, and recordable with each formal experiment.

## 2. Goals

- Remove key algorithm thresholds from production entry points.
- Preserve separate, understandable configuration domains instead of creating one giant configuration class.
- Allow CLI arguments to override versioned defaults without bypassing validation.
- Produce a resolved configuration containing the values that actually reached the algorithms.
- Save formal experiment and replay metadata with the Git revision and selected model hashes.
- Keep ordinary interactive demos convenient; they may run without producing an experiment record.
- Keep configuration modules below the repository's 500-line production-source limit.

## 3. Non-goals

- This change does not tune thresholds or claim better perception accuracy.
- It does not add datasets, training pipelines, benchmarks, or Agent policies.
- It does not place input/output paths, local credentials, UI colors, or presentation layout in the algorithm configuration.
- It does not copy model binaries into experiment directories.
- It does not replace explicit constructor injection used by unit tests and ablation experiments.

## 4. Configuration boundaries

### 4.1 Cabin configuration

`modules/config/cabin.yaml` remains the versioned source for:

- driver-presence timing;
- EAR and MAR thresholds;
- blink and yawn timing;
- PERCLOS window and readiness;
- driver-state transition thresholds.

`modules/config/cabin.py` continues to expose immutable typed objects and validates both individual ranges and relationships between state thresholds.

### 4.2 Perception configuration

`modules/config/perception.yaml` contains three focused domains:

- `phone`: detector confidence and image size, spatial-association timing, and missing-observation tolerance;
- `driving`: working resolution, object confidence, NMS IoU, warmup count, execution-provider preference, and standalone object-detector defaults;
- `lane`: smoothing, slope bounds, HLS color bounds, Canny thresholds, Hough thresholds, and road-ROI proportions.

`modules/config/perception.py` owns the immutable public aggregate. Small domain dataclasses may be placed in focused files under `modules/config/` if the aggregate approaches the size limit. Validation remains close to each domain type.

Model paths, video paths, output paths, camera indices, and `--show` stay in CLI/application code. They identify resources and runtime presentation rather than algorithm behavior.

## 5. Resolution and override rules

The effective configuration is resolved once at the application boundary:

```text
versioned YAML defaults
        ↓
typed parsing and validation
        ↓
explicit CLI overrides
        ↓
the same validation again
        ↓
immutable resolved configuration
        ↓
algorithm constructors
```

Precedence is:

1. an explicitly supplied CLI value;
2. the versioned YAML value;
3. failure if a required value is absent.

Argparse options that override configuration use `default=None`, so the resolver can distinguish “not supplied” from a deliberate false or zero value. Entry points must not duplicate YAML defaults in parser declarations.

Algorithm classes continue accepting explicit parameters for tests and ablations. Application entry points are responsible for passing resolved values, and must not embed numeric algorithm thresholds directly.

## 6. Experiment snapshots

`modules/config/snapshot.py` provides the reusable snapshot writer. `scripts/snapshot_experiment_config.py` exposes it as a CLI for validation and manual workflows. A formal evaluation or replay command will call the same library rather than launch the script as a subprocess.

A snapshot is written to a run directory as `resolved_config.yaml` and contains:

- schema version;
- run ID and UTC creation time;
- Git commit and dirty-worktree state;
- resolved cabin and perception values;
- explicit CLI overrides;
- selected model asset IDs, expected paths, sizes, and SHA-256 values copied from the asset manifest;
- a SHA-256 digest of the canonical resolved configuration.

Snapshots use deterministic key ordering for hashing and an atomic temporary-file replacement for final output. Existing snapshots are not silently overwritten.

Formal experiment mode rejects a dirty Git worktree by default. An explicit `--allow-dirty` escape hatch may be used for debugging, but the snapshot must record `dirty: true`; such runs are not suitable as final résumé evidence.

Ordinary camera or visualization demos do not have to create snapshots. Any command described as an evaluation, benchmark, ablation, or reproducible replay must create a snapshot successfully before processing data.

## 7. Error handling

Configuration loading fails before model or camera initialization when it encounters:

- an unknown section or field;
- a missing required value;
- a wrong scalar or collection type;
- a value outside its valid range;
- contradictory lower/upper or suspected/drowsy thresholds;
- invalid image dimensions, frame counts, or durations.

Errors identify the full field path, for example `perception.driving.nms_threshold`. CLI overrides pass through the same constructors and therefore receive the same errors.

Snapshot creation fails rather than producing a partial record when Git metadata, required model-manifest entries, hashing, serialization, or final atomic replacement fails.

## 8. Integration points

The first implementation updates these boundaries:

- cabin camera and video demos continue using `CabinPerceptionConfig`;
- phone demo receives detector and temporal-association parameters from the resolved perception configuration;
- integrated driving scene, standalone object, and classic lane demos receive their defaults from the resolved perception configuration;
- reusable perception services accept injected resolved values without reading CLI state;
- future context/event/Agent evaluation receives the same snapshot metadata, but Agent runtime behavior is not changed in this phase.

This keeps the dependency direction one-way: configuration may initialize perception, while configuration never imports perception, context, events, Agent, or tools.

## 9. Testing strategy

Unit tests cover:

- repository defaults loading successfully;
- immutable nested configuration values;
- partial CLI overrides and precedence;
- unknown fields and wrong types;
- range and cross-field validation;
- deterministic canonical serialization and digest generation;
- clean/dirty Git metadata handling;
- duplicate-run protection and atomic snapshot output.

Integration tests cover:

- cabin, phone, scene, object, and lane entry points consuming typed configuration rather than numeric threshold literals;
- resolved values reaching the appropriate algorithm constructor;
- snapshot generation without cameras, model downloads, or online APIs;
- wheel contents including both YAML files and installed-package default loading.

CI continues to run offline Ruff linting, Ruff formatting, focused mypy, source-size enforcement, model-manifest schema validation, pytest, and coverage reporting.

## 10. Migration sequence

1. Add failing tests for perception defaults, validation, CLI precedence, and snapshots.
2. Add typed perception configuration and package the YAML resource.
3. Migrate phone, integrated driving, object, and lane entry-point defaults one domain at a time.
4. Add snapshot library and CLI.
5. Add AST/integration guards preventing threshold literals from returning to production entry points.
6. Update README and contributor commands.
7. Run local gates, independent review, remote CI, and only then update `todolist.md`.

## 11. Acceptance criteria

- No key cabin, phone-distraction, road-object, YOLOPv2, or classic-lane algorithm threshold is declared directly in a production entry point.
- YAML defaults and explicit CLI overrides produce one immutable resolved configuration.
- Invalid YAML and invalid overrides fail before hardware or model initialization with field-specific errors.
- A formal run snapshot reproduces every resolved value and identifies the exact Git commit and selected model assets.
- Snapshot generation is deterministic and fully offline.
- Packaged installs load both versioned YAML resources.
- Existing public constructor injection and documented Demo behavior remain compatible.
- All quality gates and remote CI pass before the corresponding TODO item is checked.

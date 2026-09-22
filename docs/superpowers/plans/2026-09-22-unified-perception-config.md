# Unified Perception Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace production-entry-point perception thresholds with validated versioned configuration and generate human-readable, reproducible run artifacts that make future perception and Agent results credible and easy to present.

**Architecture:** Keep cabin defaults in the existing cabin package resource and add one separate perception YAML resource with small immutable domain dataclasses. Resolve CLI overrides at application boundaries, inject the resolved values into algorithms, and use a separate snapshot module to write a machine-readable manifest plus a concise Markdown run card. The configuration layer does not import perception or Agent code.

**Tech Stack:** Python 3.13, frozen dataclasses, PyYAML, argparse, hashlib, subprocess-based Git metadata, pytest, Ruff, mypy, uv.

## Global Constraints

- Configuration modules and helpers remain at or below 500 lines; executable `*_demo.py` entry points remain at or below 300 lines.
- Model files, video paths, output paths, credentials, and UI presentation values are not algorithm configuration.
- CLI overrides use `None` as the unset value and pass through the same validation as YAML values.
- Formal run artifacts contain real configuration and provenance only; no benchmark or accuracy number is invented.
- Snapshot generation is deterministic and works without cameras, models, or online APIs.
- Existing explicit constructor injection remains supported for tests and ablations.
- Work proceeds test-first and each task ends in a focused commit.

---

### Task 1: Add immutable perception configuration

**Files:**
- Create: `modules/config/perception.yaml`
- Create: `modules/config/perception.py`
- Modify: `modules/config/__init__.py`
- Modify: `tests/config/test_cabin_config.py`
- Create: `tests/config/test_perception_config.py`
- Modify: `tests/packaging/test_wheel.py`

**Interfaces:**
- Produces: `PerceptionConfig.load(path: str | Path) -> PerceptionConfig`
- Produces: `PerceptionConfig.load_default() -> PerceptionConfig`
- Produces: frozen `PhonePerceptionConfig`, `DrivingPerceptionConfig`, and `LanePerceptionConfig`
- Consumes: packaged YAML resources under `modules/config/`

- [ ] **Step 1: Write failing tests for defaults, immutability, packaging, and validation**

```python
def test_repository_perception_config_loads() -> None:
    config = PerceptionConfig.load_default()
    assert config.phone.confidence_threshold == pytest.approx(0.35)
    assert config.driving.nms_threshold == pytest.approx(0.45)
    assert config.lane.canny_low == 60


def test_invalid_threshold_order_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "perception.yaml"
    path.write_text(
        "lane:\n  canny_low: 180\n  canny_high: 120\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lane.canny_low"):
        PerceptionConfig.load(path)


def test_perception_configuration_is_immutable() -> None:
    config = PerceptionConfig.load_default()
    with pytest.raises(FrozenInstanceError):
        config.phone.image_size = 320  # type: ignore[misc]
```

Extend the wheel test with:

```python
assert "modules/config/perception.yaml" in wheel.namelist()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run --group dev python -m pytest \
  tests/config/test_perception_config.py \
  tests/packaging/test_wheel.py -q -p no:cacheprovider
```

Expected: collection fails because `PerceptionConfig` and `perception.yaml` do not exist.

- [ ] **Step 3: Add the versioned YAML values**

Create `modules/config/perception.yaml` with these complete defaults:

```yaml
phone:
  model_name: yolo26n.pt
  confidence_threshold: 0.35
  image_size: 640
  near_duration_seconds: 0.5
  use_duration_seconds: 1.5
  missing_tolerance_seconds: 0.25

driving:
  work_width: 1280
  work_height: 720
  score_threshold: 0.30
  nms_threshold: 0.45
  prefer_coreml: true
  warmup_runs: 2
  object_model_name: yolo11n.pt
  object_confidence_threshold: 0.25
  object_image_size: 640

lane:
  smoothing: 0.75
  min_abs_slope: 0.35
  max_abs_slope: 3.0
  white_hls_lower: [0, 160, 0]
  white_hls_upper: [180, 255, 255]
  yellow_hls_lower: [10, 80, 80]
  yellow_hls_upper: [40, 255, 255]
  canny_low: 60
  canny_high: 150
  color_canny_low: 50
  color_canny_high: 120
  roi_left: 0.05
  roi_top_left: 0.42
  roi_top_right: 0.58
  roi_right: 0.95
  roi_top_height: 0.52
  hough_threshold: 40
  min_line_length: 35
  max_line_gap: 80
```

- [ ] **Step 4: Implement focused frozen dataclasses and YAML parsing**

`modules/config/perception.py` exposes the following public shape:

```python
@dataclass(frozen=True)
class PhonePerceptionConfig:
    model_name: str = "yolo26n.pt"
    confidence_threshold: float = 0.35
    image_size: int = 640
    near_duration_seconds: float = 0.5
    use_duration_seconds: float = 1.5
    missing_tolerance_seconds: float = 0.25


@dataclass(frozen=True)
class DrivingPerceptionConfig:
    work_width: int = 1280
    work_height: int = 720
    score_threshold: float = 0.30
    nms_threshold: float = 0.45
    prefer_coreml: bool = True
    warmup_runs: int = 2
    object_model_name: str = "yolo11n.pt"
    object_confidence_threshold: float = 0.25
    object_image_size: int = 640


@dataclass(frozen=True)
class LanePerceptionConfig:
    smoothing: float = 0.75
    min_abs_slope: float = 0.35
    max_abs_slope: float = 3.0
    white_hls_lower: tuple[int, int, int] = (0, 160, 0)
    white_hls_upper: tuple[int, int, int] = (180, 255, 255)
    yellow_hls_lower: tuple[int, int, int] = (10, 80, 80)
    yellow_hls_upper: tuple[int, int, int] = (40, 255, 255)
    canny_low: int = 60
    canny_high: int = 150
    color_canny_low: int = 50
    color_canny_high: int = 120
    roi_left: float = 0.05
    roi_top_left: float = 0.42
    roi_top_right: float = 0.58
    roi_right: float = 0.95
    roi_top_height: float = 0.52
    hough_threshold: int = 40
    min_line_length: int = 35
    max_line_gap: int = 80


@dataclass(frozen=True)
class PerceptionConfig:
    phone: PhonePerceptionConfig = field(default_factory=PhonePerceptionConfig)
    driving: DrivingPerceptionConfig = field(default_factory=DrivingPerceptionConfig)
    lane: LanePerceptionConfig = field(default_factory=LanePerceptionConfig)

    @classmethod
    def default_path(cls) -> Path:
        return Path(__file__).with_name("perception.yaml")

    @classmethod
    def load_default(cls) -> PerceptionConfig:
        return cls.load(cls.default_path())
```

Validate non-empty model names, thresholds in `[0, 1]`, positive image dimensions and durations, ordered slope/Canny/ROI bounds, three-element HLS values in `[0, 255]`, and unknown sections/fields. Normalize YAML HLS lists to tuples before dataclass construction. Partial YAML files inherit dataclass defaults.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run --group dev python -m pytest \
  tests/config/test_perception_config.py \
  tests/config/test_cabin_config.py \
  tests/packaging/test_wheel.py -q -p no:cacheprovider
```

Expected: all focused tests pass and the built wheel contains both YAML files.

- [ ] **Step 6: Commit Task 1**

```bash
git add modules/config tests/config tests/packaging/test_wheel.py
git commit -m "feat: add versioned perception configuration"
```

---

### Task 2: Route application thresholds through resolved configuration

**Files:**
- Create: `modules/config/overrides.py`
- Modify: `apps/cabin_demo/phone_demo.py`
- Modify: `apps/driving_demo/scene_cli.py`
- Modify: `apps/driving_demo/scene_runner.py`
- Modify: `apps/driving_demo/object_demo.py`
- Modify: `apps/driving_demo/lane_demo.py`
- Modify: `modules/driving/lane/lane_detector.py`
- Create: `tests/config/test_perception_entrypoints.py`
- Modify: `tests/driving/test_yolopv2_contract.py`

**Interfaces:**
- Consumes: `PerceptionConfig`, `PhonePerceptionConfig`, `DrivingPerceptionConfig`, and `LanePerceptionConfig`
- Produces: `resolve_overrides(config: PerceptionConfig, overrides: Mapping[str, object]) -> PerceptionConfig`
- Preserves: explicit algorithm constructor parameters and existing CLI option names

- [ ] **Step 1: Write failing tests for override precedence and entry-point wiring**

```python
def test_explicit_override_wins_without_mutating_defaults() -> None:
    original = PerceptionConfig.load_default()
    resolved = resolve_overrides(
        original,
        {"driving.score_threshold": 0.42, "phone.image_size": 320},
    )
    assert resolved.driving.score_threshold == pytest.approx(0.42)
    assert resolved.phone.image_size == 320
    assert original.driving.score_threshold == pytest.approx(0.30)


def test_invalid_override_uses_domain_validation() -> None:
    with pytest.raises(ValueError, match="driving.nms_threshold"):
        resolve_overrides(
            PerceptionConfig.load_default(),
            {"driving.nms_threshold": 1.2},
        )
```

Add an AST test that visits numeric constants inside calls to `PhoneDetector`, `PhoneBehaviorTracker`, `PanopticDrivingDetector`, `RoadObjectDetector`, and `LaneDetector` in the five application files and asserts no numeric threshold is passed directly.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run --group dev python -m pytest \
  tests/config/test_perception_entrypoints.py \
  tests/driving/test_yolopv2_contract.py -q -p no:cacheprovider
```

Expected: import or assertion failure because override resolution and entry-point wiring do not exist.

- [ ] **Step 3: Implement immutable dotted-path overrides**

`modules/config/overrides.py` accepts only these prefixes and reconstructs frozen dataclasses with `dataclasses.replace`:

```python
ALLOWED_DOMAINS = {"phone", "driving", "lane"}


def resolve_overrides(
    config: PerceptionConfig,
    overrides: Mapping[str, object],
) -> PerceptionConfig:
    resolved = config
    for path, value in overrides.items():
        domain, separator, field_name = path.partition(".")
        if not separator or domain not in ALLOWED_DOMAINS:
            raise ValueError(f"unknown perception override: {path}")
        section = getattr(resolved, domain)
        if field_name not in {item.name for item in fields(section)}:
            raise ValueError(f"unknown perception override: {path}")
        updated_section = replace(section, **{field_name: value})
        resolved = replace(resolved, **{domain: updated_section})
    return resolved
```

Each domain dataclass validates itself in `__post_init__`, so YAML and CLI values share one validation path.

- [ ] **Step 4: Migrate the five application boundaries**

- `phone_demo.py` loads `PerceptionConfig.load_default().phone` and passes its six fields.
- `scene_cli.py` uses `None` defaults for `--work-width`, `--work-height`, `--conf`, and `--nms`; after parsing, it resolves only explicitly supplied values against the driving defaults.
- `scene_runner.py` passes the resolved driving values including CoreML preference and warmup count.
- `object_demo.py` loads object-model, confidence, and image-size defaults, while existing CLI options override them.
- `lane_demo.py` passes a complete `LanePerceptionConfig` into `LaneDetector`.
- `LaneDetector` accepts `config: LanePerceptionConfig | None = None` and retains keyword-only `smoothing`, `min_abs_slope`, and `max_abs_slope` compatibility overrides. It stores the resolved config and uses it for HLS, Canny, ROI, slope, smoothing, and Hough parameters.

- [ ] **Step 5: Verify entry points and behavior**

Run:

```bash
uv run --group dev python -m pytest \
  tests/config/test_perception_entrypoints.py \
  tests/driving/test_yolopv2_contract.py \
  tests/driving/test_scene_rendering.py -q -p no:cacheprovider
uv run --group dev python -m apps.driving_demo.scene_demo --help
```

Expected: tests pass, `--help` exits zero without requiring a model file, and rendering regression tests remain unchanged.

- [ ] **Step 6: Commit Task 2**

```bash
git add modules/config/overrides.py apps modules/driving tests/config tests/driving
git commit -m "refactor: resolve perception thresholds from config"
```

---

### Task 3: Produce reproducible run manifests and result cards

**Files:**
- Create: `modules/config/snapshot.py`
- Create: `scripts/snapshot_experiment_config.py`
- Create: `tests/config/test_snapshot.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: resolved `CabinPerceptionConfig`, resolved `PerceptionConfig`, explicit override mapping, and selected entries from `assets/model_manifest.yaml`
- Produces: `build_run_snapshot(...) -> dict[str, object]`
- Produces: `write_run_artifacts(output_dir: Path, snapshot: Mapping[str, object]) -> RunArtifactPaths`
- Produces: `resolved_config.yaml` for machines and `run_card.md` for human-facing result presentation

- [ ] **Step 1: Write failing deterministic-artifact tests**

```python
def test_snapshot_digest_is_deterministic() -> None:
    first = build_run_snapshot(
        cabin=CabinPerceptionConfig.load_default(),
        perception=PerceptionConfig.load_default(),
        overrides={"driving.score_threshold": 0.42},
        assets=[],
        provenance=RunProvenance(
            run_id="demo-001",
            created_at_utc="2026-09-22T00:00:00Z",
            git_commit="a" * 40,
            dirty=False,
        ),
    )
    second = build_run_snapshot(
        cabin=CabinPerceptionConfig.load_default(),
        perception=PerceptionConfig.load_default(),
        overrides={"driving.score_threshold": 0.42},
        assets=[],
        provenance=RunProvenance(
            run_id="demo-001",
            created_at_utc="2026-09-22T00:00:00Z",
            git_commit="a" * 40,
            dirty=False,
        ),
    )
    assert first["config_sha256"] == second["config_sha256"]


def test_writer_creates_machine_and_human_artifacts(tmp_path: Path) -> None:
    snapshot = build_run_snapshot(
        cabin=CabinPerceptionConfig.load_default(),
        perception=PerceptionConfig.load_default(),
        overrides={},
        assets=[],
        provenance=RunProvenance(
            run_id="demo-001",
            created_at_utc="2026-09-22T00:00:00Z",
            git_commit="a" * 40,
            dirty=False,
        ),
    )
    paths = write_run_artifacts(tmp_path / "demo-001", snapshot)
    assert paths.manifest.name == "resolved_config.yaml"
    assert paths.run_card.name == "run_card.md"
    assert "Git commit" in paths.run_card.read_text(encoding="utf-8")
```

Also test refusal to overwrite an existing run, dirty-worktree rejection in formal mode, missing asset IDs, and absence of secret values.

- [ ] **Step 2: Run snapshot tests and verify RED**

Run:

```bash
uv run --group dev python -m pytest \
  tests/config/test_snapshot.py -q -p no:cacheprovider
```

Expected: collection fails because `modules.config.snapshot` does not exist.

- [ ] **Step 3: Implement canonical snapshot construction**

Use these public types:

```python
@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    created_at_utc: str
    git_commit: str
    dirty: bool


@dataclass(frozen=True)
class RunArtifactPaths:
    manifest: Path
    run_card: Path
```

Convert frozen dataclasses with `dataclasses.asdict`. Compute `config_sha256` from `yaml.safe_dump(resolved_sections, sort_keys=True).encode("utf-8")`. The digest covers resolved cabin/perception values and explicit overrides, not timestamps or paths.

- [ ] **Step 4: Implement atomic artifact writing and readable result card**

Write both artifacts to temporary siblings and replace their final targets only after both serialize successfully. Refuse an output directory that already contains either final artifact.

The Markdown card contains this exact structure:

```markdown
# VehicleMind Run Card

| Field | Value |
|---|---|
| Run ID | demo-001 |
| Git commit | aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa |
| Dirty worktree | no |
| Config SHA-256 | ... |

## Selected model assets

| Asset | SHA-256 | Expected path |
|---|---|---|

## Resolved configuration

- Cabin configuration: recorded in `resolved_config.yaml`
- Perception configuration: recorded in `resolved_config.yaml`
- CLI overrides: recorded in `resolved_config.yaml`
```

This card displays traceability only. Metric tables and qualitative examples will be added by later benchmark tasks when real results exist.

- [ ] **Step 5: Add the offline CLI**

`scripts/snapshot_experiment_config.py` supports:

```text
--run-id RUN_ID
--output-root PATH
--asset-id ID        repeatable
--override PATH=VALUE repeatable
--allow-dirty
```

It loads repository defaults, resolves overrides, reads selected model metadata from the existing asset manifest, obtains `git rev-parse HEAD` and `git status --porcelain`, rejects dirty formal runs unless `--allow-dirty` is supplied, and prints both created paths.

- [ ] **Step 6: Verify artifacts in a temporary directory**

Run:

```bash
uv run --group dev python -m pytest tests/config/test_snapshot.py -q
uv run --group dev python scripts/snapshot_experiment_config.py \
  --run-id engineering-baseline \
  --output-root /tmp/vehiclemind-runs \
  --allow-dirty
```

Expected: tests pass; the command creates `/tmp/vehiclemind-runs/engineering-baseline/resolved_config.yaml` and `run_card.md` without cameras, models, or network access.

- [ ] **Step 7: Commit Task 3**

```bash
git add modules/config/snapshot.py scripts/snapshot_experiment_config.py tests/config/test_snapshot.py .gitignore
git commit -m "feat: record reproducible run artifacts"
```

---

### Task 4: Document, verify, review, and update progress

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `todolist.md`
- Modify: `docs/superpowers/plans/2026-09-22-unified-perception-config.md`

**Interfaces:**
- Produces: commands a reviewer can copy to inspect configuration and create a run card
- Produces: checked progress state only after local and remote evidence exists

- [ ] **Step 1: Document configuration and result artifacts**

Add concise README commands:

```bash
uv run --group dev python scripts/snapshot_experiment_config.py \
  --run-id engineering-baseline \
  --output-root runs
```

Explain that `resolved_config.yaml` is the machine-readable provenance record and `run_card.md` is the human-facing artifact intended for result packages. State explicitly that it contains no accuracy or latency claim until evaluation tasks populate those results.

- [ ] **Step 2: Run the complete local quality gate**

Run:

```bash
uv sync --frozen --group dev
uv run --group dev ruff check .
uv run --group dev ruff format --check apps modules scripts tests
uv run --group dev mypy \
  modules/config \
  modules/vehicle_ai/context/enums.py \
  modules/vehicle_ai/integration/cabin_adapter.py \
  scripts/verify_assets.py \
  scripts/snapshot_experiment_config.py
uv run --group dev python scripts/check_source_size.py
uv run --group dev python scripts/verify_assets.py --schema-only
uv run --group dev python -m pytest \
  -m "not hardware and not online" \
  --cov=modules \
  --cov-report=term-missing -q
git diff --check
```

Expected: every command exits zero, all tests pass, and coverage is reported without claiming the final 80% goal.

- [ ] **Step 3: Request independent review and fix all Critical/Important findings**

Review range: the commit before Task 1 through the current Task 3 head. Review configuration validation, CLI override compatibility, wheel resources, snapshot determinism, secret leakage, source sizes, and whether the run card could be mistaken for benchmark evidence.

- [ ] **Step 4: Commit documentation and evidence-backed progress update**

Only after Step 2 and Step 3 succeed, mark “将硬编码阈值迁移至可版本化配置文件，并保存每次实验使用的配置” complete.

```bash
git add README.md CONTRIBUTING.md todolist.md \
  docs/superpowers/plans/2026-09-22-unified-perception-config.md
git commit -m "docs: explain reproducible result artifacts"
```

- [ ] **Step 5: Push and verify protected remote CI**

Run:

```bash
git push
gh pr checks 1 --repo CuitSuanNaiGaiGai/VehicleMind --watch --interval 10
```

Expected: the protected `core-quality` check completes successfully on PR #1.

# VehicleMind Engineering Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current demo repository into a reproducible, testable, reviewable baseline without changing the intended perception or Agent behavior.

**Architecture:** Keep `apps/` as thin runnable entry points, move reusable configuration and validation into focused modules, and separate core dependencies from heavyweight perception dependencies. CI exercises deterministic core logic without cameras, model weights, or online LLM calls; hardware-dependent paths remain explicit local checks.

**Tech Stack:** Python 3.13, pytest, pytest-cov, Ruff, mypy, uv, PyYAML, GitHub Actions, ONNX Runtime, MediaPipe, OpenCV.

## Global Constraints

- Development happens on `codex/engineering-baseline`, never directly on `main`.
- Preserve the current macOS Apple Silicon path and CPUExecutionProvider fallback.
- Do not commit `.env`, model weights, videos, datasets, generated outputs, or local caches.
- Behavioral changes follow RED → GREEN → REFACTOR; configuration and documentation-only changes are verified with parsers and smoke commands.
- New production modules should stay below 500 lines and runnable entry points below 300 lines.
- Existing oversized files are reduced through extraction, not copied into new monoliths.
- Online LLMs, cameras, and model weights are not required by the default CI job.

---

### Task 1: Establish deterministic test collection and dependency metadata

**Files:**
- Create: `pyproject.toml`
- Create: `requirements/core.txt`
- Create: `requirements/perception.txt`
- Create: `requirements/dev.txt`
- Modify: `requirements.txt`
- Rename: `apps/cabin_demo/phone_test.py` → `apps/cabin_demo/phone_demo.py`
- Rename: `apps/vehicle_ai_demo/real_perception_test.py` → `apps/vehicle_ai_demo/real_perception_demo.py`
- Test: `tests/test_repository_layout.py`

**Interfaces:**
- Consumes: Python 3.13 and the existing `apps`, `modules`, and `tests` packages.
- Produces: `pytest` collection restricted to `tests/`, dependency groups, and unambiguous demo filenames.

- [x] **Step 1: Add a failing repository-layout test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_modules_are_not_named_like_tests():
    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "apps").rglob("*_test.py")
    )
    assert offenders == []
```

- [x] **Step 2: Verify RED**

Run: `python -m pytest tests/test_repository_layout.py -q -p no:cacheprovider`

Expected: FAIL listing `phone_test.py` and `real_perception_test.py`.

- [x] **Step 3: Rename the demo modules and add project metadata**

`pyproject.toml` must set `requires-python = ">=3.13,<3.14"`, configure pytest with `testpaths = ["tests"]`, and configure Ruff for Python 3.13. Dependency files must separate lightweight runtime, perception, and development packages; `requirements.txt` remains the full-install compatibility entry point.

- [x] **Step 4: Generate and validate the lock**

Run: `uv lock`

Expected: `uv.lock` resolves successfully for Python 3.13.

- [x] **Step 5: Verify GREEN**

Run: `uv run --group dev python -m pytest tests/test_repository_layout.py -q -p no:cacheprovider`

Expected: `1 passed`.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock requirements.txt requirements apps tests/test_repository_layout.py
git commit -m "build: establish reproducible Python baseline"
```

### Task 2: Repair stale demo contracts and unknown observation semantics

**Files:**
- Modify: `apps/vehicle_ai_demo/cabin_adapter_demo.py`
- Modify: `apps/vehicle_ai_demo/runtime_demo.py`
- Modify: `modules/cabin/perception_service.py`
- Modify: `modules/vehicle_ai/context/models.py`
- Test: `tests/vehicle_ai/test_cabin_contracts.py`

**Interfaces:**
- Consumes: `CabinContextAdapter.update(*, presence, driver_state, risk, ...)`.
- Produces: one tri-state observation contract: `eye_closed: bool | None`; demo callers use semantic `presence` rather than frame-level `face_present`.

- [x] **Step 1: Write failing contract tests**

```python
from modules.vehicle_ai.context import ContextManager, DriverPresence
from modules.vehicle_ai.integration import CabinContextAdapter


def test_adapter_accepts_semantic_presence():
    manager = ContextManager()
    adapter = CabinContextAdapter(manager)
    adapter.update(presence="PRESENT", driver_state="NORMAL", risk="LOW")
    assert manager.get_context().driver.presence is DriverPresence.PRESENT


def test_missing_eye_observation_remains_unknown():
    manager = ContextManager()
    adapter = CabinContextAdapter(manager)
    adapter.update(
        presence="PRESENT",
        driver_state="UNKNOWN",
        risk="UNKNOWN",
        eye_closed=None,
    )
    assert manager.get_context().driver.eye_closed is None
```

- [x] **Step 2: Verify the stale demos fail before the fix**

Run: `uv run --group dev python -m apps.vehicle_ai_demo.cabin_adapter_demo`

Expected: FAIL with `unexpected keyword argument 'face_present'`.

- [x] **Step 3: Apply the minimal contract repair**

Replace the two stale demo arguments with `presence="PRESENT"`; change `CabinPerceptionSnapshot.eye_closed` to `bool | None`; keep `DriverContext.eye_closed` at `bool | None = None`.

- [x] **Step 4: Verify GREEN**

Run: `uv run --group dev python -m pytest tests/vehicle_ai/test_cabin_contracts.py -q -p no:cacheprovider`

Expected: all tests pass.

Run: `uv run --group dev python -m apps.vehicle_ai_demo.cabin_adapter_demo`

Expected: exits successfully and prints NORMAL and DROWSY context transitions.

- [x] **Step 5: Commit**

```bash
git add apps/vehicle_ai_demo modules/cabin/perception_service.py modules/vehicle_ai/context/models.py tests
git commit -m "fix: align cabin demos with semantic context contract"
```

### Task 3: Replace hard-coded cabin thresholds with validated configuration

**Files:**
- Create: `modules/config/__init__.py`
- Create: `modules/config/cabin.py`
- Modify: `configs/cabin.yaml`
- Modify: `modules/cabin/perception_service.py`
- Test: `tests/config/test_cabin_config.py`

**Interfaces:**
- Produces: `CabinPerceptionConfig.load(path: str | Path) -> CabinPerceptionConfig` and nested immutable threshold dataclasses.
- Consumes: YAML containing `presence`, `eye`, `blink`, `perclos`, `mouth`, `yawn`, and `driver_state` mappings.

- [ ] **Step 1: Write failing configuration tests**

```python
from pathlib import Path

import pytest

from modules.config.cabin import CabinPerceptionConfig


def test_repository_cabin_config_loads():
    config = CabinPerceptionConfig.load(Path("configs/cabin.yaml"))
    assert config.eye.ear_threshold == pytest.approx(0.21)
    assert config.perclos.window_seconds == pytest.approx(30.0)


def test_invalid_perclos_threshold_is_rejected(tmp_path):
    path = tmp_path / "cabin.yaml"
    path.write_text("perclos:\n  window_seconds: -1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="window_seconds"):
        CabinPerceptionConfig.load(path)
```

- [ ] **Step 2: Verify RED**

Run: `uv run --group dev python -m pytest tests/config/test_cabin_config.py -q`

Expected: FAIL because `modules.config.cabin` does not exist.

- [ ] **Step 3: Implement focused immutable configuration types**

Each nested dataclass validates only its own fields in `__post_init__`. `CabinPerceptionService.__init__` accepts `config: CabinPerceptionConfig | None = None`; `None` loads the repository defaults without changing existing threshold values.

- [ ] **Step 4: Verify GREEN**

Run: `uv run --group dev python -m pytest tests/config/test_cabin_config.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/cabin.yaml modules/config modules/cabin/perception_service.py tests/config
git commit -m "refactor: centralize cabin perception configuration"
```

### Task 4: Enforce readable file boundaries and remove empty placeholders

**Files:**
- Create: `scripts/check_source_size.py`
- Create: `modules/driving/perception/types.py`
- Create: `modules/driving/perception/runtime.py`
- Create: `modules/driving/perception/preprocess.py`
- Create: `modules/driving/perception/postprocess.py`
- Modify: `modules/driving/perception/panoptic_detector.py`
- Create: focused drawing modules under `apps/cabin_demo/ui/` and `apps/driving_demo/ui/`
- Modify: oversized demo entry points to import focused helpers
- Delete: `modules/cabin/common/types.py`
- Delete: `modules/cabin/face/detector.py`
- Delete: `modules/cabin/fatigue/detector.py`
- Delete: `modules/cabin/fatigue/state_machine.py`
- Test: `tests/test_source_size.py`
- Test: `tests/driving/test_yolopv2_contract.py`

**Interfaces:**
- Produces: a source-size check with maximum 500 lines for `modules/**/*.py` and 300 lines for `apps/**/*_demo.py` entry points.
- Preserves: `PanopticDrivingDetector.detect(frame) -> DrivingSceneResult` and all documented CLI commands.

- [ ] **Step 1: Write the failing size-policy test**

```python
from scripts.check_source_size import find_oversized_sources


def test_source_files_respect_size_policy():
    assert find_oversized_sources() == []
```

- [ ] **Step 2: Verify RED**

Run: `uv run --group dev python -m pytest tests/test_source_size.py -q`

Expected: FAIL listing the existing oversized modules and demo entry points.

- [ ] **Step 3: Extract by responsibility without changing public APIs**

Move dataclasses, ONNX session creation, preprocessing, decoding, and visual rendering into focused files. Keep entry points responsible only for argument parsing, service construction, loop orchestration, and shutdown. Remove only zero-byte modules that have no importers.

- [ ] **Step 4: Verify behavior and size GREEN**

Run: `uv run --group dev python scripts/check_source_size.py`

Expected: exit 0 with no oversized files.

Run: `uv run --group dev python -m pytest tests/test_source_size.py tests/driving/test_yolopv2_contract.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps modules scripts tests
git commit -m "refactor: split oversized perception and demo modules"
```

### Task 5: Add environment, asset, licensing, and contributor documentation

**Files:**
- Create: `.env.example`
- Create: `assets/model_manifest.yaml`
- Create: `scripts/verify_assets.py`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `CONTRIBUTING.md`
- Modify: `README.md`
- Modify: `.gitignore`
- Test: `tests/test_asset_manifest.py`

**Interfaces:**
- Produces: `verify_assets(manifest_path: Path, root: Path) -> list[str]`, returning deterministic validation errors.
- Requires: owner-selected project license before creating `LICENSE`; third-party provenance must be evidence-backed.

- [ ] **Step 1: Write failing manifest tests**

```python
from pathlib import Path

from scripts.verify_assets import verify_assets


def test_missing_assets_have_actionable_errors(tmp_path):
    errors = verify_assets(Path("assets/model_manifest.yaml"), tmp_path)
    assert errors
    assert all("expected path" in error for error in errors)
```

- [ ] **Step 2: Verify RED**

Run: `uv run --group dev python -m pytest tests/test_asset_manifest.py -q`

Expected: FAIL because the verifier does not exist.

- [ ] **Step 3: Add manifest and verification implementation**

The manifest records the existing MediaPipe and YOLOPv2 filenames, SHA-256 values, canonical upstream URL, license URL, and local expected path. `.env.example` contains key names only. README documents supported platforms, CPU fallback, installation groups, asset verification, and valid demo commands.

- [ ] **Step 4: Verify documentation and manifest**

Run: `uv run --group dev python -m pytest tests/test_asset_manifest.py -q`

Expected: all tests pass.

Run: `uv run --group dev python scripts/verify_assets.py --manifest assets/model_manifest.yaml`

Expected: either success for locally present assets or actionable missing-asset messages without a traceback.

- [ ] **Step 5: Commit non-license documentation**

```bash
git add .env.example .gitignore assets/model_manifest.yaml scripts/verify_assets.py THIRD_PARTY_NOTICES.md CONTRIBUTING.md README.md tests/test_asset_manifest.py
git commit -m "docs: document environment and third-party assets"
```

### Task 6: Add deterministic CI and offline smoke checks

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/smoke/test_core_demos.py`
- Modify: `pyproject.toml`

**Interfaces:**
- CI installs only the core and development dependency groups.
- Smoke tests run deterministic context, selector, event, tool, and cabin-adapter demos without cameras, weights, or API keys.

- [ ] **Step 1: Add a failing subprocess smoke test for every core demo**

```python
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "apps.vehicle_ai_demo.context_demo",
        "apps.vehicle_ai_demo.context_manager_demo",
        "apps.vehicle_ai_demo.context_selector_demo",
        "apps.vehicle_ai_demo.event_demo",
        "apps.vehicle_ai_demo.tool_demo",
        "apps.vehicle_ai_demo.cabin_adapter_demo",
    ],
)
def test_core_demo_exits_successfully(module):
    result = subprocess.run(
        [sys.executable, "-m", module],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Verify RED against the stale baseline, then GREEN after Task 2**

Run: `uv run --group dev python -m pytest tests/smoke/test_core_demos.py -q`

Expected after Task 2: all cases pass.

- [ ] **Step 3: Add CI jobs**

CI runs `ruff check`, `mypy` on the focused core packages, pytest with coverage, the source-size check, and asset-manifest schema validation. Hardware-dependent perception tests are explicitly marked and excluded from the default job.

- [ ] **Step 4: Verify locally**

Run: `uv run --group dev ruff check .`

Run: `uv run --group dev python -m pytest -q`

Expected: both commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add .github pyproject.toml tests/smoke
git commit -m "ci: add deterministic core quality gates"
```

### Task 7: Update progress, push the branch, open a PR, then protect main

**Files:**
- Modify: `todolist.md`

**Interfaces:**
- Produces: checked TODO items only where evidence exists, a remote feature branch, and a reviewable PR.

- [ ] **Step 1: Run the complete verification suite**

Run: `uv sync --group dev`

Run: `uv run --group dev ruff check .`

Run: `uv run --group dev python -m pytest --cov=modules --cov-report=term-missing -q`

Run: `uv run --group dev python scripts/check_source_size.py`

Expected: all commands exit 0; core coverage is reported and no source-size violations remain.

- [ ] **Step 2: Update `todolist.md` conservatively**

Mark only requirements proven by the verification output. Leave `LICENSE` unchecked until the repository owner selects a license, and leave branch protection unchecked until the GitHub rule is confirmed by API readback.

- [ ] **Step 3: Push and create a Pull Request**

```bash
git push -u origin codex/engineering-baseline
gh pr create --base main --head codex/engineering-baseline --title "Establish VehicleMind engineering baseline" --body-file docs/superpowers/plans/2026-09-22-engineering-baseline.md
```

- [ ] **Step 4: Enable and verify main protection after CI exists**

Require pull requests and the exact CI check names introduced in `.github/workflows/ci.yml`; query the protection endpoint afterward and save the result in the handoff report.

---

## Self-Review Result

- Scope covers every item in `todolist.md` section 1 except selecting the project license, which requires the repository owner's legal preference.
- The source-size task directly addresses the explicit requirement to avoid thousand-line and multi-thousand-line files.
- Public contracts remain `CabinContextAdapter.update`, `CabinPerceptionService.process_frame`, and `PanopticDrivingDetector.detect`; extractions do not change their callers.
- Hardware and online services are excluded from default CI but remain documented local verification paths.

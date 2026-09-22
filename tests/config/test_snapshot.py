from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from modules.config import CabinPerceptionConfig, PerceptionConfig
from modules.config.overrides import resolve_overrides
from modules.config.snapshot import (
    RunProvenance,
    build_run_snapshot,
    select_asset_records,
    write_run_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _provenance(*, dirty: bool = False) -> RunProvenance:
    return RunProvenance(
        run_id="demo-001",
        created_at_utc="2026-09-22T00:00:00Z",
        git_commit="a" * 40,
        dirty=dirty,
    )


def _snapshot(*, dirty: bool = False, allow_dirty: bool = False):
    overrides = {"driving.score_threshold": 0.42}
    return build_run_snapshot(
        cabin=CabinPerceptionConfig.load_default(),
        perception=resolve_overrides(PerceptionConfig.load_default(), overrides),
        overrides=overrides,
        assets=[],
        provenance=_provenance(dirty=dirty),
        allow_dirty=allow_dirty,
    )


def test_snapshot_digest_is_deterministic() -> None:
    first = _snapshot()
    second = _snapshot()

    assert first["config_sha256"] == second["config_sha256"]
    assert len(str(first["config_sha256"])) == 64


def test_dirty_formal_run_is_rejected() -> None:
    with pytest.raises(ValueError, match="dirty worktree"):
        _snapshot(dirty=True)


def test_dirty_debug_run_is_recorded_when_explicitly_allowed() -> None:
    snapshot = _snapshot(dirty=True, allow_dirty=True)

    assert snapshot["provenance"]["dirty"] is True


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "nested/run", "nested\\run", ".", "..", "line\nbreak"],
)
def test_run_id_must_be_a_safe_directory_name(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        RunProvenance(
            run_id=run_id,
            created_at_utc="2026-09-22T00:00:00Z",
            git_commit="a" * 40,
            dirty=False,
        )


def test_writer_creates_machine_and_human_artifacts(tmp_path: Path) -> None:
    paths = write_run_artifacts(tmp_path / "demo-001", _snapshot())

    document = yaml.safe_load(paths.manifest.read_text(encoding="utf-8"))
    run_card = paths.run_card.read_text(encoding="utf-8")
    assert paths.manifest.name == "resolved_config.yaml"
    assert paths.run_card.name == "run_card.md"
    assert document["provenance"]["git_commit"] == "a" * 40
    assert "# VehicleMind Run Card" in run_card
    assert "Git commit" in run_card
    assert "2026-09-22T00:00:00Z" in run_card
    assert "Driving score threshold" in run_card
    assert "0.42" in run_card
    assert "No benchmark metrics are recorded" in run_card


def test_writer_refuses_to_overwrite_run_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo-001"
    write_run_artifacts(output_dir, _snapshot())

    with pytest.raises(FileExistsError, match="already exists"):
        write_run_artifacts(output_dir, _snapshot())


def test_writer_refuses_any_preexisting_run_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo-001"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_run_artifacts(output_dir, _snapshot())


def test_selected_assets_preserve_traceability_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "assets": [
                    {
                        "id": "model-a",
                        "expected_path": "models/model-a.onnx",
                        "size_bytes": 12,
                        "sha256": "b" * 64,
                        "source_url": "https://example.invalid/model-a",
                        "license": {
                            "name": "test-only",
                            "url": "https://example.invalid/license",
                            "status": "test-only",
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    records = select_asset_records(manifest_path, ["model-a"])

    assert records == [
        {
            "id": "model-a",
            "expected_path": "models/model-a.onnx",
            "size_bytes": 12,
            "sha256": "b" * 64,
        }
    ]


def test_unknown_selected_asset_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("version: 1\nassets: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown model asset id: missing"):
        select_asset_records(manifest_path, ["missing"])


def test_snapshot_cli_runs_offline_without_models_or_cameras(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/snapshot_experiment_config.py",
            "--run-id",
            "cli-smoke",
            "--output-root",
            str(tmp_path),
            "--allow-dirty",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli-smoke/resolved_config.yaml").is_file()
    assert (tmp_path / "cli-smoke/run_card.md").is_file()

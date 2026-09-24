from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.perception_audit.runner import run_audit


def setup_assets(tmp_path: Path):
    cabin, road = tmp_path / "private-cabin", tmp_path / "private-road"
    cabin.mkdir()
    road.mkdir()
    models = {"cabin": tmp_path / "face.task", "road": tmp_path / "road.onnx"}
    for model in models.values():
        model.write_bytes(b"test model")
    return cabin, road, models


def fake_catalog(_cabin, _road):
    return {"attempted": {"cabin": 1, "road": 1}, "items": [
        {"id": "cabin/a.mp4", "domain": "cabin", "basename": "a.mp4",
         "sha256": "a" * 64, "probe_status": "ok"},
        {"id": "road/b.mp4", "domain": "road", "basename": "b.mp4",
         "sha256": "b" * 64, "probe_status": "cannot_open"},
    ]}


def fake_process(_catalog, _dirs, **_kwargs):
    return [
        {"id": "cabin/a.mp4", "domain": "cabin", "status": "success",
         "processed_frames": 3, "valid_output_frames": 3,
         "state_counts": {"NORMAL": 3}, "face_visible_frames": 3,
         "eye_closed_frames": 0, "yawn_output_frames": 0},
        {"id": "road/b.mp4", "domain": "road", "status": "failed",
         "error": "视频无法打开", "processed_frames": 0, "valid_output_frames": 0},
    ]


def test_runner_writes_private_artifacts_and_complete_marker_last(tmp_path: Path):
    cabin, road, models = setup_assets(tmp_path)
    output = run_audit(
        cabin, road, tmp_path / "runs", run_id="test-1", model_paths=models,
        catalog_factory=fake_catalog, process_factory=fake_process,
    )
    assert (output / ".complete").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "report.html").is_file()
    assert len(list((output / "videos").glob("*.json"))) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["run_id"] == "test-1"
    assert manifest["provenance"]["models"]["cabin"]["sha256"]
    assert "git" in manifest["provenance"]
    assert "dependencies" in manifest["provenance"]
    effective = manifest["provenance"]["configuration"]["effective"]
    assert effective["cabin"]["eye"]["ear_threshold"] == 0.21
    assert effective["road"]["work_width"] == 1280
    assert summary["accuracy"]["status"] == "not_evaluated"
    assert summary["domains"]["road"]["videos"] == {
        "successful": 0, "attempted": 1
    }
    for artifact in output.rglob("*"):
        if artifact.is_file():
            content = artifact.read_text(encoding="utf-8")
            assert str(tmp_path) not in content
            assert "api-key" not in content


def test_runner_refuses_to_overwrite_old_run(tmp_path: Path):
    cabin, road, models = setup_assets(tmp_path)
    output_root = tmp_path / "runs"
    output_root.mkdir()
    old = output_root / "test-1"
    old.mkdir()
    (old / "keep.txt").write_text("unchanged")
    with pytest.raises(FileExistsError):
        run_audit(cabin, road, output_root, run_id="test-1", model_paths=models,
                  catalog_factory=fake_catalog, process_factory=fake_process)
    assert (old / "keep.txt").read_text() == "unchanged"


def test_fatal_model_error_keeps_run_incomplete(tmp_path: Path):
    cabin, road, models = setup_assets(tmp_path)

    def fail(_catalog, _dirs, **_kwargs):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        run_audit(cabin, road, tmp_path / "runs", run_id="test-2",
                  model_paths=models, catalog_factory=fake_catalog,
                  process_factory=fail)
    output = tmp_path / "runs" / "test-2"
    assert (output / "manifest.json").exists()
    assert not (output / ".complete").exists()

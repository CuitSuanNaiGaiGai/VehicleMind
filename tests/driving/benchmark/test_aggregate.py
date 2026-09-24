from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_road_benchmarks import aggregate_runs


def write_run(root: Path, index: int, **changes: object) -> Path:
    directory = root / f"run-{index}"
    directory.mkdir()
    result = {
        "config": {
            "provider": "cpu",
            "work_width": 1280,
            "work_height": 720,
            "warmup_frames": 5,
            "measure_frames": 30,
            "score_threshold": 0.3,
            "nms_threshold": 0.45,
        },
        "active_providers": ["CPUExecutionProvider"],
        "provenance": {
            "video": {"sha256": "video-hash"},
            "model": {"sha256": "model-hash"},
            "git_commit": "commit-a",
            "dirty": False,
        },
        "summary": {
            "inference_ms": {"p50": 10.0 + index, "p95": 20.0 + index},
            "frame_total_ms": {"p50": 30.0 + index, "p95": 40.0 + index},
        },
        "throughput_fps": 20.0 + index,
        "environment": {
            "peak_rss_mib": 100.0 + index,
            "system": "Darwin",
            "release": "25.6.0",
            "machine": "arm64",
            "python": "3.13.9",
            "onnxruntime": "1.30.0",
            "opencv": "4.14.0",
            "numpy": "2.5.3",
        },
        "run_id": f"run-{index}",
        "process_id": 1000 + index,
    }
    for key, value in changes.items():
        if key == "model_hash":
            result["provenance"]["model"]["sha256"] = value
        elif key == "provider":
            result["config"]["provider"] = value
        elif key == "python":
            result["environment"]["python"] = value
    path = directory / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    (directory / ".complete").touch()
    return path


def test_aggregate_three_matching_runs(tmp_path: Path) -> None:
    paths = [write_run(tmp_path, index) for index in (1, 2, 3)]
    report = aggregate_runs(paths)
    assert "三次独立进程" in report
    assert "同一视频帧" in report
    assert "非感知精度" in report
    assert "21.00–23.00" in report
    assert "| 1 |" in report


@pytest.mark.parametrize("changes", [{"model_hash": "other"}, {"provider": "coreml"}])
def test_mismatched_runs_are_rejected(tmp_path: Path, changes: dict[str, str]) -> None:
    paths = [write_run(tmp_path, index) for index in (1, 2)]
    paths.append(write_run(tmp_path, 3, **changes))
    with pytest.raises(ValueError, match="不一致"):
        aggregate_runs(paths)


def test_incomplete_run_is_rejected(tmp_path: Path) -> None:
    paths = [write_run(tmp_path, index) for index in (1, 2, 3)]
    (paths[2].parent / ".complete").unlink()
    with pytest.raises(ValueError, match="未完成"):
        aggregate_runs(paths)


def test_duplicate_process_is_not_three_independent_runs(tmp_path: Path) -> None:
    paths = [write_run(tmp_path, index) for index in (1, 2, 3)]
    duplicate = json.loads(paths[0].read_text(encoding="utf-8"))
    paths[2].write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="独立进程"):
        aggregate_runs(paths)


def test_mismatched_runtime_is_rejected(tmp_path: Path) -> None:
    paths = [write_run(tmp_path, index) for index in (1, 2)]
    paths.append(write_run(tmp_path, 3, python="3.14.0"))
    with pytest.raises(ValueError, match="不一致"):
        aggregate_runs(paths)

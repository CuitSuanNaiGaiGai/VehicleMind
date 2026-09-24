from __future__ import annotations

import json

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from modules.driving.benchmark.runner import BenchmarkConfig, run_benchmark
from modules.driving.benchmark import runner
from scripts import benchmark_road_perception


class FakeCapture:
    def __init__(self, count: int) -> None:
        self.remaining = count
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.read_count += 1
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def __init__(self, providers: list[str]) -> None:
        self.session = SimpleNamespace(get_providers=lambda: providers)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return SimpleNamespace(
            preprocess_ms=1.0,
            inference_ms=2.0,
            postprocess_ms=3.0,
            total_ms=6.0,
        )


def config(tmp_path: Path, **overrides) -> BenchmarkConfig:
    video = tmp_path / "sample.mp4"
    model = tmp_path / "model.onnx"
    video.write_bytes(b"sample video")
    model.write_bytes(b"sample model")
    return BenchmarkConfig(
        video=video,
        model=model,
        provider=overrides.get("provider", "cpu"),
        work_width=4,
        work_height=4,
        warmup_frames=2,
        measure_frames=4,
        output_dir=tmp_path / "result",
    )


def test_warmup_is_excluded_and_paths_are_redacted(tmp_path: Path) -> None:
    capture = FakeCapture(7)
    detector = FakeDetector(["CPUExecutionProvider"])
    factory_kwargs = {}

    def detector_factory(**kwargs):
        factory_kwargs.update(kwargs)
        return detector

    result = run_benchmark(
        config(tmp_path),
        detector_factory=detector_factory,
        capture_factory=lambda path: capture,
    )

    assert len(result["samples"]) == 4
    assert result["summary"]["inference_ms"]["count"] == 4
    assert capture.read_count == 6
    assert capture.released is True
    assert detector.calls == 6
    assert result["provenance"]["video"]["name"] == "sample.mp4"
    assert result["provenance"]["model"]["name"] == "model.onnx"
    assert result["active_providers"] == ["CPUExecutionProvider"]
    assert isinstance(result["run_id"], str) and result["run_id"]
    assert isinstance(result["process_id"], int) and result["process_id"] > 0
    assert factory_kwargs["coreml_cache_dir"] == tmp_path / ".coreml_cache"
    assert not (tmp_path / ".coreml_cache").exists()
    assert (tmp_path / "result" / ".complete").is_file()
    report = (tmp_path / "result" / "report.md").read_text(encoding="utf-8")
    serialized = (tmp_path / "result" / "result.json").read_text(encoding="utf-8")
    assert json.loads(serialized)["samples"] == result["samples"]
    assert str(tmp_path) not in serialized + report
    assert "不含视频绘制或编码" in report


def test_short_video_leaves_no_result(tmp_path: Path) -> None:
    capture = FakeCapture(5)
    with pytest.raises(ValueError, match="视频帧不足"):
        run_benchmark(
            config(tmp_path),
            detector_factory=lambda **kwargs: FakeDetector(["CPUExecutionProvider"]),
            capture_factory=lambda path: capture,
        )
    assert capture.released is True
    assert not (tmp_path / "result").exists()


def test_coreml_request_must_activate_coreml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CoreML"):
        run_benchmark(
            config(tmp_path, provider="coreml"),
            detector_factory=lambda **kwargs: FakeDetector(["CPUExecutionProvider"]),
            capture_factory=lambda path: FakeCapture(7),
        )
    assert not (tmp_path / "result").exists()


def test_existing_result_is_not_overwritten(tmp_path: Path) -> None:
    requested = config(tmp_path)
    requested.output_dir.mkdir()
    sentinel = requested.output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_benchmark(
            requested,
            detector_factory=lambda **kwargs: FakeDetector(["CPUExecutionProvider"]),
            capture_factory=lambda path: FakeCapture(7),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_report_error_removes_only_its_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = config(tmp_path)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def fail_report(result):
        raise RuntimeError("render failed")

    monkeypatch.setattr(runner, "render_report", fail_report)
    with pytest.raises(RuntimeError, match="render failed"):
        run_benchmark(
            requested,
            detector_factory=lambda **kwargs: FakeDetector(["CPUExecutionProvider"]),
            capture_factory=lambda path: FakeCapture(7),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not requested.output_dir.exists()
    assert not list(tmp_path.glob(".result.tmp-*"))


def test_cli_hides_path_in_unexpected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_run(config):
        raise LookupError(f"private path: {tmp_path}")

    monkeypatch.setattr(benchmark_road_perception, "run_benchmark", fail_run)
    code = benchmark_road_perception.main(
        [
            "--video",
            str(tmp_path / "sample.mp4"),
            "--model",
            str(tmp_path / "model.onnx"),
            "--provider",
            "cpu",
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    assert code == 2
    error_text = capsys.readouterr().err
    assert "LookupError" in error_text
    assert str(tmp_path) not in error_text


def test_racing_existing_directory_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = config(tmp_path)
    original_render = runner.render_report

    def create_competing_directory(result):
        requested.output_dir.mkdir()
        return original_render(result)

    monkeypatch.setattr(runner, "render_report", create_competing_directory)
    with pytest.raises(FileExistsError):
        run_benchmark(
            requested,
            detector_factory=lambda **kwargs: FakeDetector(["CPUExecutionProvider"]),
            capture_factory=lambda path: FakeCapture(7),
        )
    assert requested.output_dir.is_dir()
    assert list(requested.output_dir.iterdir()) == []

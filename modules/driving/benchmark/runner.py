"""Bounded, no-encoding road-perception benchmark over local video frames."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable

import cv2
import numpy as np

from modules.driving.benchmark.report import STAGES, render_report
from modules.driving.benchmark.stats import summarize_ms, throughput_fps


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BenchmarkConfig:
    video: Path
    model: Path
    provider: Literal["cpu", "coreml"]
    output_dir: Path
    work_width: int = 1280
    work_height: int = 720
    warmup_frames: int = 5
    measure_frames: int = 30


def _default_detector_factory(**kwargs: Any) -> Any:
    from modules.driving.perception.panoptic_detector import PanopticDrivingDetector

    return PanopticDrivingDetector(**kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path) -> dict[str, str | int]:
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _validate(config: BenchmarkConfig) -> None:
    for name in ("work_width", "work_height", "warmup_frames", "measure_frames"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if config.provider not in ("cpu", "coreml"):
        raise ValueError("provider 必须为 cpu 或 coreml")
    if not config.video.is_file():
        raise ValueError(f"视频不存在：{config.video.name}")
    if not config.model.is_file():
        raise ValueError(f"模型不存在：{config.model.name}")
    if config.output_dir.exists():
        raise FileExistsError(f"结果目录已存在：{config.output_dir.name}")


def _read_and_detect(
    capture: Any, detector: Any, config: BenchmarkConfig
) -> dict[str, float]:
    frame_start = time.perf_counter()
    success, frame = capture.read()
    read_end = time.perf_counter()
    if not success or frame is None:
        raise ValueError("视频帧不足，无法完成指定预热与测量")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("视频帧必须是 BGR 三通道图像")

    resize_start = time.perf_counter()
    if frame.shape[1] != config.work_width or frame.shape[0] != config.work_height:
        working = cv2.resize(
            frame,
            (config.work_width, config.work_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = frame
    resize_end = time.perf_counter()
    detection = detector.detect(working)
    frame_end = time.perf_counter()
    return {
        "video_read_ms": (read_end - frame_start) * 1000,
        "resize_ms": (resize_end - resize_start) * 1000,
        "preprocess_ms": float(detection.preprocess_ms),
        "inference_ms": float(detection.inference_ms),
        "postprocess_ms": float(detection.postprocess_ms),
        "detector_total_ms": float(detection.total_ms),
        "frame_total_ms": (frame_end - frame_start) * 1000,
    }


def _write_result(destination: Path, result: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    published: list[Path] = []
    reserved = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "report.md").write_text(render_report(result), encoding="utf-8")
        (staging / ".complete").write_text("", encoding="utf-8")
        destination.mkdir()
        reserved = True
        for name in ("result.json", "report.md", ".complete"):
            target = destination / name
            os.link(staging / name, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            if target.exists() and os.path.samefile(target, staging / target.name):
                target.unlink()
        if reserved:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(staging)


def run_benchmark(
    config: BenchmarkConfig,
    detector_factory: Callable[..., Any] | None = None,
    capture_factory: Callable[[str], Any] | None = None,
) -> dict[str, object]:
    """Measure one local run and publish exclusively with a completion marker."""

    _validate(config)
    cache_dir = config.output_dir.parent / ".coreml_cache"
    cache_existed = cache_dir.exists()
    factory = detector_factory or _default_detector_factory
    started = time.perf_counter()
    detector = factory(
        model_path=config.model,
        prefer_coreml=config.provider == "coreml",
        warmup_runs=0,
        coreml_cache_dir=cache_dir,
    )
    session_init_ms = (time.perf_counter() - started) * 1000
    active_providers = list(detector.session.get_providers())
    if (
        config.provider == "coreml"
        and "CoreMLExecutionProvider" not in active_providers
    ):
        raise ValueError("CoreML 未实际启用，不能将 CPU 回退记为 CoreML 成绩")
    if config.provider == "cpu" and "CoreMLExecutionProvider" in active_providers:
        raise ValueError("CPU 模式实际启用了 CoreML，请检查 provider 配置")

    opener = capture_factory or cv2.VideoCapture
    capture = opener(str(config.video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"无法打开视频：{config.video.name}")
    samples: list[dict[str, float]] = []
    first_frame_ms = 0.0
    measurement_started = 0.0
    measurement_ended = 0.0
    try:
        for index in range(config.warmup_frames + config.measure_frames):
            if index == config.warmup_frames:
                measurement_started = time.perf_counter()
            sample = _read_and_detect(capture, detector, config)
            if index == 0:
                first_frame_ms = sample["frame_total_ms"]
            if index >= config.warmup_frames:
                samples.append(sample)
                measurement_ended = time.perf_counter()
    finally:
        capture.release()

    summary = {
        key: summarize_ms([sample[key] for sample in samples]) for key, _ in STAGES
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "config": {
            "provider": config.provider,
            "work_width": config.work_width,
            "work_height": config.work_height,
            "warmup_frames": config.warmup_frames,
            "measure_frames": config.measure_frames,
            "score_threshold": 0.30,
            "nms_threshold": 0.45,
        },
        "active_providers": active_providers,
        "provenance": {
            "video": _asset(config.video),
            "model": _asset(config.model),
            "git_commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "coreml_cache_existed": cache_existed,
        },
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "onnxruntime": _package_version("onnxruntime"),
            "peak_rss_mib": _peak_rss_mib(),
        },
        "timing": {
            "session_init_ms": session_init_ms,
            "cold_first_frame_ms": first_frame_ms,
            "measurement_wall_seconds": measurement_ended - measurement_started,
        },
        "throughput_fps": throughput_fps(
            len(samples), measurement_ended - measurement_started
        ),
        "samples": samples,
        "summary": summary,
    }
    _write_result(config.output_dir, result)
    return result

"""Run a reproducible local audit and persist only structured evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from modules.perception_audit.catalog import scan_catalog
from modules.perception_audit.process import process_catalog
from modules.perception_audit.report import build_summary, render_html, render_markdown
from modules.config import CabinPerceptionConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = {
    "cabin": REPOSITORY_ROOT / "models/mediapipe/face_landmarker.task",
    "road": REPOSITORY_ROOT / "models/driving/YOLOPv2_512.onnx",
}
ROAD_SERVICE_CONFIG = {
    "work_width": 1280, "work_height": 720, "score_threshold": 0.30,
    "nms_threshold": 0.45, "prefer_coreml": True, "warmup_runs": 2,
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size,
            "sha256": _hash_file(path)}


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _provenance(model_paths: dict[str, Path], sample_interval: int) -> dict:
    configs = {}
    for name in ("cabin.yaml", "perception.yaml"):
        path = REPOSITORY_ROOT / "modules/config" / name
        configs[name] = _asset(path)
    status = _git(["status", "--porcelain"])
    return {
        "git": {"commit": _git(["rev-parse", "HEAD"]),
                "dirty": None if status is None else bool(status)},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "dependencies": {name: _version(name) for name in
                         ("numpy", "opencv-python", "mediapipe", "onnxruntime")},
        "models": {domain: _asset(path) for domain, path in model_paths.items()},
        "configuration": {"sample_interval": sample_interval,
                          "mode": "full_video_no_frame_skip",
                          "config_files": configs,
                          "effective": {
                              "cabin": asdict(CabinPerceptionConfig.load_default()),
                              "road": ROAD_SERVICE_CONFIG,
                          }},
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _model_factory(model_paths: dict[str, Path]):
    def create(domain: str):
        if domain == "cabin":
            from modules.cabin.perception_service import CabinPerceptionService

            return CabinPerceptionService(
                model_paths[domain], config=CabinPerceptionConfig.load_default())
        if domain == "road":
            from modules.driving.perception_service import DrivingPerceptionService

            return DrivingPerceptionService(model_paths[domain], **ROAD_SERVICE_CONFIG)
        raise ValueError(f"不支持的感知域: {domain}")

    return create


def run_audit(
    cabin_dir: Path,
    road_dir: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    model_paths: dict[str, Path] | None = None,
    sample_interval: int = 30,
    catalog_factory: Callable = scan_catalog,
    process_factory: Callable = process_catalog,
) -> Path:
    """Create a new immutable run directory; `.complete` is written last."""
    if sample_interval < 1:
        raise ValueError("sample_interval 必须大于 0")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id 包含不支持的字符")
    models = {domain: Path(path) for domain, path in
              (model_paths or DEFAULT_MODELS).items()}
    if set(models) != {"cabin", "road"}:
        raise ValueError("必须提供舱内和舱外两个模型")
    for path in models.values():
        if not path.is_file():
            raise FileNotFoundError(f"模型不存在：{path.name}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / run_id
    output.mkdir(exist_ok=False)
    catalog = catalog_factory(Path(cabin_dir), Path(road_dir))
    manifest = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog": catalog,
        "provenance": _provenance(models, sample_interval),
    }
    _write_json(output / "manifest.json", manifest)
    results = process_factory(
        catalog, {"cabin": Path(cabin_dir), "road": Path(road_dir)},
        service_factory=_model_factory(models), sample_interval=sample_interval,
    )
    if [row["id"] for row in results] != [item["id"] for item in catalog["items"]]:
        raise ValueError("处理结果与冻结清单不一致")
    video_dir = output / "videos"
    video_dir.mkdir()
    for index, result in enumerate(results):
        _write_json(video_dir / f"{index:03d}.json", result)
    summary = build_summary(catalog, results)
    _write_json(output / "summary.json", summary)
    (output / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    (output / "report.html").write_text(render_html(summary), encoding="utf-8")
    (output / ".complete").write_text("complete\n", encoding="utf-8")
    return output

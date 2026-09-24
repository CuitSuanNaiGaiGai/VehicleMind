"""Freeze a small local video inventory without inventing ground truth."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import cv2

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v"})
MAX_VIDEOS_PER_DOMAIN = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(path: Path, capture_factory: Callable) -> dict:
    capture = None
    try:
        capture = capture_factory(str(path))
        if not capture.isOpened():
            return {"probe_status": "cannot_open", "probe_error": "视频无法打开"}
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            return {"probe_status": "empty_video", "probe_error": "视频元数据无有效帧"}
        decoded, frame = capture.read()
        if not decoded or frame is None:
            return {"probe_status": "decode_failed", "probe_error": "首帧解码失败"}
        return {
            "probe_status": "ok",
            "probe_error": None,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
        }
    except Exception as exc:
        return {"probe_status": "probe_error", "probe_error": type(exc).__name__}
    finally:
        if capture is not None:
            capture.release()


def scan_catalog(
    cabin_dir: Path,
    road_dir: Path,
    *,
    capture_factory: Callable = cv2.VideoCapture,
) -> dict:
    """Return a JSON-safe, deterministically ordered inventory for both domains.

    Paths are used only for probing and hashing; no absolute path enters the result.
    A failed probe stays in the attempted count. Reading the returned list freezes
    the run input set even if files are added later.
    """
    domains = {"cabin": Path(cabin_dir), "road": Path(road_dir)}
    frozen: dict[str, list[Path]] = {}
    for domain, directory in domains.items():
        if not directory.is_dir():
            raise ValueError(f"{domain} 输入目录不存在")
        files = sorted(
            (path for path in directory.iterdir() if path.is_file()
             and path.suffix.lower() in VIDEO_SUFFIXES),
            key=lambda path: path.name,
        )
        if not files:
            raise ValueError(f"{domain} 输入目录没有视频")
        if len(files) > MAX_VIDEOS_PER_DOMAIN:
            raise ValueError(f"{domain} 视频数超过 {MAX_VIDEOS_PER_DOMAIN}")
        frozen[domain] = files

    seen_hashes: dict[str, str] = {}
    items: list[dict] = []
    for domain, paths in frozen.items():
        for path in paths:
            video_id = f"{domain}/{path.name}"
            digest = _sha256(path)
            item = {
                "id": video_id,
                "domain": domain,
                "basename": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "duplicate_of": seen_hashes.get(digest),
                "source_status": "unknown",
                "license_status": "unknown",
                "label_status": "unknown",
            }
            seen_hashes.setdefault(digest, video_id)
            item.update(_probe(path, capture_factory))
            items.append(item)

    return {
        "attempted": {domain: len(paths) for domain, paths in frozen.items()},
        "items": items,
    }

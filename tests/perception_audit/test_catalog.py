from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modules.perception_audit.catalog import scan_catalog


class FakeCapture:
    def __init__(self, opened: bool = True, frames: int = 30, decodes: bool = True) -> None:
        self.opened = opened
        self.frames = frames
        self.decodes = decodes
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        import cv2

        return {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_COUNT: self.frames,
        }.get(property_id, 0)

    def release(self) -> None:
        self.released = True

    def read(self):
        return self.decodes, object() if self.decodes else None


def write_video(directory: Path, name: str, content: bytes = b"video") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def test_sorted_frozen_catalog_and_unknown_truth(tmp_path: Path) -> None:
    cabin, road = tmp_path / "cabin", tmp_path / "road"
    write_video(cabin, "b.mp4", b"same")
    write_video(cabin, "a.MP4", b"same")
    write_video(road, "z.mov", b"road")
    (cabin / "readme.txt").write_text("not video")
    captures: list[FakeCapture] = []

    def factory(_path: str) -> FakeCapture:
        capture = FakeCapture()
        captures.append(capture)
        return capture

    result = scan_catalog(cabin, road, capture_factory=factory)
    assert [item["id"] for item in result["items"]] == [
        "cabin/a.MP4", "cabin/b.mp4", "road/z.mov"
    ]
    assert result["attempted"] == {"cabin": 2, "road": 1}
    first = result["items"][0]
    assert first["sha256"] == hashlib.sha256(b"same").hexdigest()
    assert first["duplicate_of"] is None
    assert result["items"][1]["duplicate_of"] == first["id"]
    assert first["bytes"] == 4
    assert first["width"] == 640 and first["height"] == 480
    assert first["fps"] == 10 and first["frame_count"] == 30
    assert first["duration_seconds"] == 3
    assert first["source_status"] == "unknown"
    assert first["license_status"] == "unknown"
    assert first["label_status"] == "unknown"
    assert all(capture.released for capture in captures)
    assert str(tmp_path) not in str(result)


def test_probe_failures_remain_in_attempted_denominator(tmp_path: Path) -> None:
    cabin, road = tmp_path / "cabin", tmp_path / "road"
    write_video(cabin, "bad.mp4")
    write_video(road, "empty.mp4")
    captures = [FakeCapture(opened=False), FakeCapture(frames=0)]
    result = scan_catalog(cabin, road, capture_factory=lambda _path: captures.pop(0))
    assert result["attempted"] == {"cabin": 1, "road": 1}
    assert [item["probe_status"] for item in result["items"]] == [
        "cannot_open", "empty_video"
    ]


def test_undecodable_video_is_not_accepted(tmp_path: Path) -> None:
    cabin, road = tmp_path / "cabin", tmp_path / "road"
    write_video(cabin, "bad.mp4")
    write_video(road, "good.mp4")
    captures = [FakeCapture(decodes=False), FakeCapture()]
    result = scan_catalog(cabin, road, capture_factory=lambda _path: captures.pop(0))
    assert result["items"][0]["probe_status"] == "decode_failed"
    assert result["attempted"] == {"cabin": 1, "road": 1}


@pytest.mark.parametrize("domain", ["cabin", "road"])
def test_rejects_more_than_50_without_truncating(tmp_path: Path, domain: str) -> None:
    cabin, road = tmp_path / "cabin", tmp_path / "road"
    write_video(cabin, "one.mp4")
    write_video(road, "one.mp4")
    for index in range(50):
        write_video(tmp_path / domain, f"{index:02}.mp4")
    with pytest.raises(ValueError, match="50"):
        scan_catalog(cabin, road, capture_factory=lambda _path: FakeCapture())


def test_rejects_empty_domain(tmp_path: Path) -> None:
    cabin, road = tmp_path / "cabin", tmp_path / "road"
    cabin.mkdir()
    write_video(road, "one.mp4")
    with pytest.raises(ValueError, match="cabin"):
        scan_catalog(cabin, road)

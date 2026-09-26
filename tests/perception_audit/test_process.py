from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import time

import cv2
import pytest

from modules.perception_audit.process import process_catalog, process_video


class FakeCapture:
    def __init__(self, timestamps: list[float], *, fail_at: int | None = None):
        self.timestamps = timestamps
        self.fail_at = fail_at
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.fail_at is not None and self.index == self.fail_at:
            raise OSError("broken decoder")
        if self.index >= len(self.timestamps):
            return False, None
        self.index += 1
        return True, object()

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[self.index - 1]
        return 0

    def release(self):
        self.released = True


class FakeCabinService:
    def __init__(self):
        self.timestamps = []
        self.closed = False

    def process_frame(self, _frame, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        state = "NORMAL" if len(self.timestamps) < 3 else "DROWSY"
        return SimpleNamespace(
            metadata=SimpleNamespace(valid=True, processing_ms=len(self.timestamps)),
            face_visible=True,
            presence="PRESENT",
            driver_state=state,
            risk="LOW",
            perclos=0.2,
            perclos_ready=True,
            eye_closed=False,
            current_yawn=False,
            blink_count=0,
            recent_yawns=0,
        )

    def close(self):
        self.closed = True


class FakeRoadService:
    def __init__(self):
        self.timestamps = []
        self.closed = False
        self.detector = SimpleNamespace(
            session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        )

    def process_frame(self, _frame, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return SimpleNamespace(
            metadata=SimpleNamespace(valid=True),
            scene_result=SimpleNamespace(objects=[SimpleNamespace(class_name="car")]),
            lane_detected=len(self.timestamps) % 2 == 1,
            drivable_area_detected=True,
            drivable_ratio=0.4,
            total_objects=1,
            vehicle_count=1,
            pedestrian_count=0,
            rider_count=0,
            traffic_light_count=0,
            traffic_sign_count=0,
        )

    def close(self):
        self.closed = True


def entry(domain: str, name: str, frames: int = 3) -> dict:
    return {
        "id": f"{domain}/{name}",
        "domain": domain,
        "basename": name,
        "probe_status": "ok",
        "frame_count": frames,
        "fps": 10,
    }


def test_cabin_processes_all_frames_at_native_timestamps(tmp_path: Path):
    capture = FakeCapture([0, 100, 200])
    service = FakeCabinService()
    result = process_video(
        entry("cabin", "one.mp4"),
        tmp_path,
        capture_factory=lambda _: capture,
        service_factory=lambda _: service,
        sample_interval=2,
    )
    assert result["status"] == "success"
    assert result["processed_frames"] == result["valid_output_frames"] == 3
    assert service.timestamps == [0, 100, 200]
    assert result["state_counts"] == {"NORMAL": 2, "DROWSY": 1}
    assert result["state_duration_seconds"] == pytest.approx(
        {"NORMAL": 0.2, "DROWSY": 0.1}
    )
    assert result["transitions"] == [
        {
            "frame_index": 2,
            "timestamp_ms": 200,
            "field": "driver_state",
            "from": "NORMAL",
            "to": "DROWSY",
        }
    ]
    assert [point["frame_index"] for point in result["samples"]] == [0, 2]
    assert capture.released and service.closed
    assert str(tmp_path) not in str(result)
    assert "frame" not in result["samples"][0]
    assert result["performance"]["first_frame_ms"] == 1
    assert result["performance"]["steady_frame_ms_samples"] == [2, 3]
    assert result["performance"]["service_init_ms"] >= 0
    assert result["performance"]["replay_loop_ms"] >= 0


def test_road_counts_outputs_and_adjacent_flips(tmp_path: Path):
    result = process_video(
        entry("road", "one.mp4"),
        tmp_path,
        capture_factory=lambda _: FakeCapture([0, 100, 200]),
        service_factory=lambda _: FakeRoadService(),
    )
    assert result["object_counts"] == {"car": 3}
    assert result["lane_detected_frames"] == 2
    assert result["drivable_detected_frames"] == 3
    assert result["lane_output_flips"] == 2
    assert result["drivable_output_flips"] == 0
    assert result["active_providers"] == ["CPUExecutionProvider"]
    assert "performance" not in result


def test_cabin_timing_excludes_initialization_and_close_from_replay(
    tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])

    class TimedCapture(FakeCapture):
        def read(self):
            clock[0] += 0.01
            return super().read()

        def release(self):
            clock[0] += 0.02
            super().release()

    class TimedService(FakeCabinService):
        def close(self):
            clock[0] += 10
            super().close()

    def service_factory(_domain):
        clock[0] += 0.5
        return TimedService()

    def capture_factory(_path):
        clock[0] += 0.03
        return TimedCapture([0])

    result = process_video(
        entry("cabin", "one.mp4", frames=1),
        tmp_path,
        capture_factory=capture_factory,
        service_factory=service_factory,
    )
    assert result["performance"] == pytest.approx(
        {
            "service_init_ms": 500,
            "first_frame_ms": 1,
            "steady_frame_ms_samples": [],
            "replay_loop_ms": 70,
        }
    )


def test_cabin_invalid_output_still_records_service_processing_time(tmp_path):
    class InvalidCabinService(FakeCabinService):
        def process_frame(self, frame, timestamp_ms):
            snapshot = super().process_frame(frame, timestamp_ms)
            snapshot.metadata.valid = False
            return snapshot

    result = process_video(
        entry("cabin", "invalid.mp4", frames=2),
        tmp_path,
        capture_factory=lambda _: FakeCapture([0, 100]),
        service_factory=lambda _: InvalidCabinService(),
    )
    assert result["status"] == "success"
    assert result["processed_frames"] == 2
    assert result["valid_output_frames"] == 0
    assert result["performance"]["first_frame_ms"] == 1
    assert result["performance"]["steady_frame_ms_samples"] == [2]


def test_capture_creation_failure_remains_failed_and_closes_service(tmp_path):
    service = FakeCabinService()

    def broken_capture(_path):
        raise OSError("capture unavailable")

    result = process_video(
        entry("cabin", "bad.mp4"),
        tmp_path,
        capture_factory=broken_capture,
        service_factory=lambda _: service,
    )
    assert result["status"] == "failed"
    assert result["performance"]["first_frame_ms"] is None
    assert result["performance"]["steady_frame_ms_samples"] == []
    assert service.closed


def test_timestamp_fallback_is_explicit_when_codec_has_no_clock(tmp_path: Path):
    service = FakeCabinService()
    result = process_video(
        entry("cabin", "one.mp4"),
        tmp_path,
        capture_factory=lambda _: FakeCapture([0, 0, 0]),
        service_factory=lambda _: service,
    )
    assert service.timestamps == [0, 100, 200]
    assert result["timestamp_fallback_frames"] == 2


def test_each_video_gets_new_service_and_failure_does_not_stop_batch(tmp_path):
    captures = iter([FakeCapture([0, 100], fail_at=1), FakeCapture([0, 100])])
    services = []

    def factory(_domain):
        service = FakeCabinService()
        services.append(service)
        return service

    catalog = {"items": [entry("cabin", "bad.mp4", 2), entry("cabin", "good.mp4", 2)]}
    results = process_catalog(
        catalog,
        {"cabin": tmp_path},
        capture_factory=lambda _: next(captures),
        service_factory=factory,
    )
    assert [result["status"] for result in results] == ["failed", "success"]
    assert results[0]["processed_frames"] == 1
    assert results[1]["processed_frames"] == 2
    assert len(services) == 2 and all(service.closed for service in services)
    assert str(tmp_path) not in str(results)


def test_model_initialization_failure_is_fatal(tmp_path):
    with pytest.raises(RuntimeError, match="model unavailable"):
        process_catalog(
            {"items": [entry("cabin", "one.mp4")]},
            {"cabin": tmp_path},
            capture_factory=lambda _: FakeCapture([0]),
            service_factory=lambda _: (_ for _ in ()).throw(
                RuntimeError("model unavailable")
            ),
        )


def test_probe_failure_stays_in_batch_results(tmp_path):
    failed = entry("road", "broken.mp4") | {
        "probe_status": "cannot_open",
        "probe_error": "视频无法打开",
    }
    result = process_catalog(
        {"items": [failed]},
        {"road": tmp_path},
        capture_factory=lambda _: pytest.fail("should not open"),
        service_factory=lambda _: pytest.fail("should not init model"),
    )
    assert result[0]["status"] == "failed"
    assert result[0]["error"] == "视频无法打开"
    assert result[0]["processed_frames"] == 0


def test_container_frame_count_is_estimate_not_failure(tmp_path):
    result = process_video(
        entry("cabin", "short.mp4", frames=3),
        tmp_path,
        capture_factory=lambda _: FakeCapture([0, 100]),
        service_factory=lambda _: FakeCabinService(),
    )
    assert result["status"] == "success"
    assert result["processed_frames"] == 2
    assert result["frame_count_difference"] == -1


def test_invalid_output_breaks_adjacent_transition_chain(tmp_path):
    class GapRoadService(FakeRoadService):
        def process_frame(self, frame, timestamp_ms):
            snapshot = super().process_frame(frame, timestamp_ms)
            if len(self.timestamps) == 2:
                snapshot.metadata.valid = False
            if len(self.timestamps) == 3:
                snapshot.lane_detected = False
            return snapshot

    result = process_video(
        entry("road", "gap.mp4"),
        tmp_path,
        capture_factory=lambda _: FakeCapture([0, 100, 200]),
        service_factory=lambda _: GapRoadService(),
    )
    assert result["valid_output_frames"] == 2
    assert result["lane_output_flips"] == 0
    assert result["transitions"] == []


def test_modified_video_is_rejected_before_model_initialization(tmp_path):
    video = tmp_path / "changed.mp4"
    video.write_bytes(b"new content")
    frozen_hash = hashlib.sha256(b"old content").hexdigest()
    result = process_video(
        entry("cabin", "changed.mp4", frames=1) | {"sha256": frozen_hash},
        tmp_path,
        capture_factory=lambda _: pytest.fail("should not decode"),
        service_factory=lambda _: pytest.fail("should not load model"),
    )
    assert result["status"] == "failed"
    assert result["error"] == "文件与冻结清单不一致"


def test_video_changed_while_processing_is_not_reported_success(tmp_path):
    video = tmp_path / "during.mp4"
    video.write_bytes(b"original")
    frozen_hash = hashlib.sha256(b"original").hexdigest()

    class MutatingService(FakeCabinService):
        def process_frame(self, frame, timestamp_ms):
            video.write_bytes(b"changed")
            return super().process_frame(frame, timestamp_ms)

    result = process_video(
        entry("cabin", "during.mp4", frames=1) | {"sha256": frozen_hash},
        tmp_path,
        capture_factory=lambda _: FakeCapture([0]),
        service_factory=lambda _: MutatingService(),
    )
    assert result["status"] == "failed"
    assert result["error"] == "文件与冻结清单不一致"

"""Run existing perception services on complete, isolated local videos."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2


def default_service_factory(domain: str):
    """Construct one fresh model-backed service for one video."""
    root = Path(__file__).resolve().parents[2]
    if domain == "cabin":
        from modules.cabin.perception_service import CabinPerceptionService

        return CabinPerceptionService(root / "models/mediapipe/face_landmarker.task")
    if domain == "road":
        from modules.driving.perception_service import DrivingPerceptionService

        return DrivingPerceptionService(root / "models/driving/YOLOPv2_512.onnx")
    raise ValueError(f"不支持的感知域: {domain}")


def _name(value) -> str:
    return str(getattr(value, "value", value))


def _timestamp_ms(capture, frame_index: int, fps: float, previous: int) -> tuple[int, bool]:
    raw = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if math.isfinite(raw) and raw >= 0 and (frame_index == 0 or raw > previous):
        return int(raw), False
    # Some codecs expose 0/NaN for every frame. Mark the derived fallback explicitly.
    return max(previous + 1, round(frame_index * 1000 / fps)), True


def _cabin_observation(snapshot) -> dict:
    return {
        "face_visible": bool(snapshot.face_visible),
        "presence": _name(snapshot.presence),
        "driver_state": _name(snapshot.driver_state),
        "risk": _name(snapshot.risk),
        "eye_closed": snapshot.eye_closed,
        "current_yawn": bool(snapshot.current_yawn),
        "perclos": snapshot.perclos,
        "perclos_ready": bool(snapshot.perclos_ready),
        "blink_count": int(snapshot.blink_count),
        "recent_yawns": int(snapshot.recent_yawns),
    }


def _road_observation(snapshot) -> dict:
    return {
        "object_count": int(snapshot.total_objects),
        "vehicle_count": int(snapshot.vehicle_count),
        "pedestrian_count": int(snapshot.pedestrian_count),
        "rider_count": int(snapshot.rider_count),
        "traffic_light_count": int(snapshot.traffic_light_count),
        "traffic_sign_count": int(snapshot.traffic_sign_count),
        "lane_detected": bool(snapshot.lane_detected),
        "drivable_area_detected": bool(snapshot.drivable_area_detected),
        "drivable_ratio": float(snapshot.drivable_ratio),
    }


def _initial_result(item: dict) -> dict:
    result = {
        "id": item["id"],
        "domain": item["domain"],
        "status": "failed",
        "error": None,
        "processed_frames": 0,
        "valid_output_frames": 0,
        "timestamp_fallback_frames": 0,
        "samples": [],
        "transitions": [],
    }
    if item["domain"] == "cabin":
        result.update(state_counts={}, face_visible_frames=0,
                      eye_closed_frames=0, yawn_output_frames=0)
    else:
        result.update(object_counts={}, lane_detected_frames=0,
                      drivable_detected_frames=0, lane_output_flips=0,
                      drivable_output_flips=0)
    return result


def _record_output(result: dict, domain: str, observation: dict,
                   previous: dict | None, frame_index: int,
                   timestamp_ms: int, sample_interval: int,
                   snapshot) -> None:
    if frame_index % sample_interval == 0:
        result["samples"].append({"frame_index": frame_index,
                                  "timestamp_ms": timestamp_ms,
                                  "output": observation})
    if domain == "cabin":
        state = observation["driver_state"]
        counts = result["state_counts"]
        counts[state] = counts.get(state, 0) + 1
        result["face_visible_frames"] += int(observation["face_visible"])
        result["eye_closed_frames"] += int(observation["eye_closed"] is True)
        result["yawn_output_frames"] += int(observation["current_yawn"])
        transition_fields = ("driver_state",)
    else:
        counts = result["object_counts"]
        for obj in snapshot.scene_result.objects:
            label = _name(obj.class_name)
            counts[label] = counts.get(label, 0) + 1
        result["lane_detected_frames"] += int(observation["lane_detected"])
        result["drivable_detected_frames"] += int(observation["drivable_area_detected"])
        if previous is not None:
            result["lane_output_flips"] += int(
                previous["lane_detected"] != observation["lane_detected"])
            result["drivable_output_flips"] += int(
                previous["drivable_area_detected"]
                != observation["drivable_area_detected"])
        transition_fields = ("lane_detected", "drivable_area_detected")
    if previous is not None:
        for field in transition_fields:
            if previous[field] != observation[field]:
                result["transitions"].append({
                    "frame_index": frame_index, "timestamp_ms": timestamp_ms,
                    "field": field, "from": previous[field], "to": observation[field],
                })


def process_video(
    item: dict,
    input_dir: Path,
    *,
    capture_factory: Callable = cv2.VideoCapture,
    service_factory: Callable = default_service_factory,
    sample_interval: int = 30,
) -> dict:
    """Process an entire video; model initialization failures propagate globally."""
    if sample_interval < 1:
        raise ValueError("sample_interval 必须大于 0")
    result = _initial_result(item)
    if item["probe_status"] != "ok":
        result["error"] = item.get("probe_error") or item["probe_status"]
        return result
    path = Path(input_dir) / item["basename"]
    service = service_factory(item["domain"])
    if item["domain"] == "road":
        session = getattr(getattr(service, "detector", None), "session", None)
        get_providers = getattr(session, "get_providers", None)
        result["active_providers"] = list(get_providers()) if callable(get_providers) else []
    capture = None
    try:
        capture = capture_factory(str(path))
        if not capture.isOpened():
            raise OSError("视频无法打开")
        previous = None
        last_timestamp = -1
        fps = float(item["fps"])
        if fps <= 0:
            raise ValueError("视频 FPS 无效")
        while True:
            decoded, frame = capture.read()
            if not decoded or frame is None:
                break
            index = result["processed_frames"]
            timestamp, fallback = _timestamp_ms(capture, index, fps, last_timestamp)
            last_timestamp = timestamp
            result["timestamp_fallback_frames"] += int(fallback)
            snapshot = service.process_frame(frame, timestamp_ms=timestamp)
            result["processed_frames"] += 1
            if not snapshot.metadata.valid:
                continue
            result["valid_output_frames"] += 1
            observation = (_cabin_observation(snapshot) if item["domain"] == "cabin"
                           else _road_observation(snapshot))
            _record_output(result, item["domain"], observation, previous,
                           index, timestamp, sample_interval, snapshot)
            previous = observation
        if result["processed_frames"] == 0:
            raise ValueError("视频无有效帧")
        if result["processed_frames"] < int(item["frame_count"]):
            raise ValueError("视频提前结束，实际帧数少于媒体元数据")
        result["status"] = "success"
    except Exception as exc:
        # Avoid leaking local paths or decoder messages into public artifacts.
        result["error"] = f"{type(exc).__name__}: 视频处理失败"
    finally:
        if capture is not None:
            capture.release()
        service.close()
    return result


def process_catalog(
    catalog: dict,
    input_dirs: dict[str, Path],
    *,
    capture_factory: Callable = cv2.VideoCapture,
    service_factory: Callable = default_service_factory,
    sample_interval: int = 30,
) -> list[dict]:
    """Continue after video failures, but not after model construction failures."""
    results = []
    for item in catalog["items"]:
        results.append(process_video(
            item, input_dirs[item["domain"]], capture_factory=capture_factory,
            service_factory=service_factory, sample_interval=sample_interval,
        ))
    return results

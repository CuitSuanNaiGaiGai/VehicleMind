from __future__ import annotations

import math

from dataclasses import dataclass, field, fields
from numbers import Real
from pathlib import Path
from typing import Any, TypeVar

import yaml


def _real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: Any) -> float:
    result = _real(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _unit_interval(name: str, value: Any) -> float:
    result = _real(name, value)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _model_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip() or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{name} must be a model identifier, not a local path")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _hls_triplet(name: str, value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must contain three integers")
    result = tuple(value)
    if any(item < 0 or item > 255 for item in result):
        raise ValueError(f"{name} values must be between zero and 255")
    return result


@dataclass(frozen=True)
class PhonePerceptionConfig:
    model_name: str = "yolo26n.pt"
    confidence_threshold: float = 0.35
    image_size: int = 640
    near_duration_seconds: float = 0.5
    use_duration_seconds: float = 1.5
    missing_tolerance_seconds: float = 0.25

    def __post_init__(self) -> None:
        _model_identifier("phone.model_name", self.model_name)
        _unit_interval("phone.confidence_threshold", self.confidence_threshold)
        _positive_int("phone.image_size", self.image_size)
        _positive("phone.near_duration_seconds", self.near_duration_seconds)
        _positive("phone.use_duration_seconds", self.use_duration_seconds)
        _positive("phone.missing_tolerance_seconds", self.missing_tolerance_seconds)


@dataclass(frozen=True)
class DrivingPerceptionConfig:
    work_width: int = 1280
    work_height: int = 720
    score_threshold: float = 0.30
    nms_threshold: float = 0.45
    prefer_coreml: bool = True
    warmup_runs: int = 2
    object_model_name: str = "yolo11n.pt"
    object_confidence_threshold: float = 0.25
    object_image_size: int = 640

    def __post_init__(self) -> None:
        _positive_int("driving.work_width", self.work_width)
        _positive_int("driving.work_height", self.work_height)
        _unit_interval("driving.score_threshold", self.score_threshold)
        _unit_interval("driving.nms_threshold", self.nms_threshold)
        if not isinstance(self.prefer_coreml, bool):
            raise ValueError("driving.prefer_coreml must be a boolean")
        _non_negative_int("driving.warmup_runs", self.warmup_runs)
        _model_identifier("driving.object_model_name", self.object_model_name)
        _unit_interval(
            "driving.object_confidence_threshold",
            self.object_confidence_threshold,
        )
        _positive_int("driving.object_image_size", self.object_image_size)


@dataclass(frozen=True)
class LanePerceptionConfig:
    smoothing: float = 0.75
    min_abs_slope: float = 0.35
    max_abs_slope: float = 3.0
    white_hls_lower: tuple[int, int, int] = (0, 160, 0)
    white_hls_upper: tuple[int, int, int] = (180, 255, 255)
    yellow_hls_lower: tuple[int, int, int] = (10, 80, 80)
    yellow_hls_upper: tuple[int, int, int] = (40, 255, 255)
    canny_low: int = 60
    canny_high: int = 150
    color_canny_low: int = 50
    color_canny_high: int = 120
    roi_left: float = 0.05
    roi_top_left: float = 0.42
    roi_top_right: float = 0.58
    roi_right: float = 0.95
    roi_top_height: float = 0.52
    fit_top_height: float = 0.55
    hough_threshold: int = 40
    min_line_length: int = 35
    max_line_gap: int = 80

    def __post_init__(self) -> None:
        _unit_interval("lane.smoothing", self.smoothing)
        minimum_slope = _positive("lane.min_abs_slope", self.min_abs_slope)
        maximum_slope = _positive("lane.max_abs_slope", self.max_abs_slope)
        if minimum_slope > maximum_slope:
            raise ValueError("lane.min_abs_slope must not exceed lane.max_abs_slope")

        for name in (
            "white_hls_lower",
            "white_hls_upper",
            "yellow_hls_lower",
            "yellow_hls_upper",
        ):
            object.__setattr__(
                self, name, _hls_triplet(f"lane.{name}", getattr(self, name))
            )
        for color in ("white", "yellow"):
            lower = getattr(self, f"{color}_hls_lower")
            upper = getattr(self, f"{color}_hls_upper")
            if any(
                lower_value > upper_value
                for lower_value, upper_value in zip(lower, upper)
            ):
                raise ValueError(
                    f"lane.{color}_hls_lower must not exceed "
                    f"lane.{color}_hls_upper component-wise"
                )

        canny_low = _positive_int("lane.canny_low", self.canny_low)
        canny_high = _positive_int("lane.canny_high", self.canny_high)
        if canny_low > canny_high:
            raise ValueError("lane.canny_low must not exceed lane.canny_high")
        color_low = _positive_int("lane.color_canny_low", self.color_canny_low)
        color_high = _positive_int("lane.color_canny_high", self.color_canny_high)
        if color_low > color_high:
            raise ValueError(
                "lane.color_canny_low must not exceed lane.color_canny_high"
            )

        roi = {
            "lane.roi_left": _unit_interval("lane.roi_left", self.roi_left),
            "lane.roi_top_left": _unit_interval("lane.roi_top_left", self.roi_top_left),
            "lane.roi_top_right": _unit_interval(
                "lane.roi_top_right", self.roi_top_right
            ),
            "lane.roi_right": _unit_interval("lane.roi_right", self.roi_right),
        }
        if not (
            roi["lane.roi_left"]
            < roi["lane.roi_top_left"]
            < roi["lane.roi_top_right"]
            < roi["lane.roi_right"]
        ):
            raise ValueError(
                "lane.roi_left, lane.roi_top_left, lane.roi_top_right, "
                "and lane.roi_right must be strictly increasing"
            )
        _unit_interval("lane.roi_top_height", self.roi_top_height)
        _unit_interval("lane.fit_top_height", self.fit_top_height)
        _positive_int("lane.hough_threshold", self.hough_threshold)
        _positive_int("lane.min_line_length", self.min_line_length)
        _positive_int("lane.max_line_gap", self.max_line_gap)


SectionConfig = TypeVar(
    "SectionConfig",
    PhonePerceptionConfig,
    DrivingPerceptionConfig,
    LanePerceptionConfig,
)


def _section(
    name: str,
    config_type: type[SectionConfig],
    value: Any,
) -> SectionConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")

    allowed = {item.name for item in fields(config_type)}
    unknown = set(value) - allowed
    if unknown:
        unknown_text = ", ".join(f"{name}.{item}" for item in sorted(unknown))
        raise ValueError(f"unknown perception configuration key: {unknown_text}")

    try:
        return config_type(**value)
    except TypeError as exc:
        raise ValueError(f"invalid {name} configuration: {exc}") from exc


@dataclass(frozen=True)
class PerceptionConfig:
    phone: PhonePerceptionConfig = field(default_factory=PhonePerceptionConfig)
    driving: DrivingPerceptionConfig = field(default_factory=DrivingPerceptionConfig)
    lane: LanePerceptionConfig = field(default_factory=LanePerceptionConfig)

    @classmethod
    def default_path(cls) -> Path:
        return Path(__file__).with_name("perception.yaml")

    @classmethod
    def load_default(cls) -> PerceptionConfig:
        return cls.load(cls.default_path())

    @classmethod
    def load(cls, path: str | Path) -> PerceptionConfig:
        config_path = Path(path)
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise ValueError("perception configuration root must be a mapping")

        section_types = {
            "phone": PhonePerceptionConfig,
            "driving": DrivingPerceptionConfig,
            "lane": LanePerceptionConfig,
        }
        unknown_sections = set(document) - set(section_types)
        if unknown_sections:
            unknown_text = ", ".join(sorted(str(item) for item in unknown_sections))
            raise ValueError(
                f"unknown perception configuration section: {unknown_text}"
            )

        return cls(
            phone=_section("phone", PhonePerceptionConfig, document.get("phone")),
            driving=_section(
                "driving", DrivingPerceptionConfig, document.get("driving")
            ),
            lane=_section("lane", LanePerceptionConfig, document.get("lane")),
        )

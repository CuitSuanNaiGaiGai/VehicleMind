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
    if not 0 < result < 1:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PresenceConfig:
    present_confirm_seconds: float = 0.15
    absence_timeout_seconds: float = 1.5
    startup_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "present_confirm_seconds",
            "absence_timeout_seconds",
            "startup_timeout_seconds",
        ):
            _positive(name, getattr(self, name))


@dataclass(frozen=True)
class EyeConfig:
    ear_threshold: float = 0.21

    def __post_init__(self) -> None:
        _unit_interval("ear_threshold", self.ear_threshold)


@dataclass(frozen=True)
class BlinkConfig:
    min_closed_frames: int = 2
    max_closed_frames: int = 15

    def __post_init__(self) -> None:
        minimum = _positive_int("min_closed_frames", self.min_closed_frames)
        maximum = _positive_int("max_closed_frames", self.max_closed_frames)
        if minimum > maximum:
            raise ValueError("min_closed_frames must not exceed max_closed_frames")


@dataclass(frozen=True)
class PerclosConfig:
    window_seconds: float = 30.0
    min_observation_seconds: float = 5.0

    def __post_init__(self) -> None:
        window = _positive("window_seconds", self.window_seconds)
        observation = _positive("min_observation_seconds", self.min_observation_seconds)
        if observation > window:
            raise ValueError("min_observation_seconds must not exceed window_seconds")


@dataclass(frozen=True)
class MouthConfig:
    mar_threshold: float = 0.35

    def __post_init__(self) -> None:
        _positive("mar_threshold", self.mar_threshold)


@dataclass(frozen=True)
class YawnConfig:
    min_open_seconds: float = 1.2

    def __post_init__(self) -> None:
        _positive("min_open_seconds", self.min_open_seconds)


@dataclass(frozen=True)
class DriverStateConfig:
    suspected_perclos: float = 0.25
    drowsy_perclos: float = 0.30
    suspected_closure_seconds: float = 1.2
    drowsy_closure_seconds: float = 2.0
    yawn_window_seconds: float = 60.0
    suspected_yawns: int = 2

    def __post_init__(self) -> None:
        suspected_perclos = _unit_interval("suspected_perclos", self.suspected_perclos)
        drowsy_perclos = _unit_interval("drowsy_perclos", self.drowsy_perclos)
        if suspected_perclos > drowsy_perclos:
            raise ValueError("suspected_perclos must not exceed drowsy_perclos")

        suspected_closure = _positive(
            "suspected_closure_seconds", self.suspected_closure_seconds
        )
        drowsy_closure = _positive(
            "drowsy_closure_seconds", self.drowsy_closure_seconds
        )
        if suspected_closure > drowsy_closure:
            raise ValueError(
                "suspected_closure_seconds must not exceed drowsy_closure_seconds"
            )

        _positive("yawn_window_seconds", self.yawn_window_seconds)
        _positive_int("suspected_yawns", self.suspected_yawns)


SectionConfig = TypeVar(
    "SectionConfig",
    PresenceConfig,
    EyeConfig,
    BlinkConfig,
    PerclosConfig,
    MouthConfig,
    YawnConfig,
    DriverStateConfig,
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
        unknown_text = ", ".join(sorted(str(item) for item in unknown))
        raise ValueError(f"unknown {name} configuration key: {unknown_text}")

    try:
        return config_type(**value)
    except TypeError as exc:
        raise ValueError(f"invalid {name} configuration: {exc}") from exc


@dataclass(frozen=True)
class CabinPerceptionConfig:
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    perclos: PerclosConfig = field(default_factory=PerclosConfig)
    mouth: MouthConfig = field(default_factory=MouthConfig)
    yawn: YawnConfig = field(default_factory=YawnConfig)
    driver_state: DriverStateConfig = field(default_factory=DriverStateConfig)

    @classmethod
    def default_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "configs/cabin.yaml"

    @classmethod
    def load_default(cls) -> CabinPerceptionConfig:
        return cls.load(cls.default_path())

    @classmethod
    def load(cls, path: str | Path) -> CabinPerceptionConfig:
        config_path = Path(path)
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise ValueError("cabin configuration root must be a mapping")

        section_types = {
            "presence": PresenceConfig,
            "eye": EyeConfig,
            "blink": BlinkConfig,
            "perclos": PerclosConfig,
            "mouth": MouthConfig,
            "yawn": YawnConfig,
            "driver_state": DriverStateConfig,
        }
        unknown_sections = set(document) - set(section_types)
        if unknown_sections:
            unknown_text = ", ".join(sorted(str(item) for item in unknown_sections))
            raise ValueError(f"unknown cabin configuration section: {unknown_text}")

        return cls(
            presence=_section("presence", PresenceConfig, document.get("presence")),
            eye=_section("eye", EyeConfig, document.get("eye")),
            blink=_section("blink", BlinkConfig, document.get("blink")),
            perclos=_section("perclos", PerclosConfig, document.get("perclos")),
            mouth=_section("mouth", MouthConfig, document.get("mouth")),
            yawn=_section("yawn", YawnConfig, document.get("yawn")),
            driver_state=_section(
                "driver_state",
                DriverStateConfig,
                document.get("driver_state"),
            ),
        )

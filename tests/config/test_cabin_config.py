from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from modules.config.cabin import CabinPerceptionConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_repository_cabin_config_loads() -> None:
    config = CabinPerceptionConfig.load(
        REPOSITORY_ROOT / "configs/cabin.yaml"
    )

    assert config.eye.ear_threshold == pytest.approx(0.21)
    assert config.perclos.window_seconds == pytest.approx(30.0)
    assert config.driver_state.drowsy_perclos == pytest.approx(0.30)


def test_partial_config_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cabin.yaml"
    path.write_text("eye:\n  ear_threshold: 0.24\n", encoding="utf-8")

    config = CabinPerceptionConfig.load(path)

    assert config.eye.ear_threshold == pytest.approx(0.24)
    assert config.perclos.window_seconds == pytest.approx(30.0)


def test_invalid_perclos_window_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cabin.yaml"
    path.write_text("perclos:\n  window_seconds: -1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="window_seconds"):
        CabinPerceptionConfig.load(path)


def test_invalid_driver_state_order_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cabin.yaml"
    path.write_text(
        "driver_state:\n"
        "  suspected_perclos: 0.40\n"
        "  drowsy_perclos: 0.30\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="suspected_perclos"):
        CabinPerceptionConfig.load(path)


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cabin.yaml"
    path.write_text("eye:\n  magic_threshold: 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="magic_threshold"):
        CabinPerceptionConfig.load(path)


def test_configuration_is_immutable() -> None:
    config = CabinPerceptionConfig()

    with pytest.raises(FrozenInstanceError):
        config.eye.ear_threshold = 0.5  # type: ignore[misc]

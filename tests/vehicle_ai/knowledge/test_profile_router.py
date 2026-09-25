from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modules.vehicle_ai.knowledge.profile_router import ProfileRouter


def _config(tmp_path: Path, **overrides: object) -> Path:
    config: dict[str, object] = {
        "instances": {
            "vehicle_common": "http://127.0.0.1:9621",
            "vehiclemind_demo": "http://127.0.0.1:9622",
        },
        "timeout_seconds": 12,
        "top_k": 5,
    }
    config.update(overrides)
    path = tmp_path / "knowledge.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_router_resolves_demo_to_its_fixed_endpoint(tmp_path: Path) -> None:
    router = ProfileRouter(_config(tmp_path))

    assert router.resolve("vehiclemind_demo") == (
        "vehiclemind_demo",
        "http://127.0.0.1:9622",
    )


@pytest.mark.parametrize("profile", [None, "", "other_car", "http://evil.example"])
def test_router_unknown_profile_only_uses_common(
    tmp_path: Path, profile: str | None
) -> None:
    router = ProfileRouter(_config(tmp_path))

    assert router.resolve(profile) == ("vehicle_common", "http://127.0.0.1:9621")


def test_router_rejects_extra_profile_endpoint(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        instances={
            "vehicle_common": "http://127.0.0.1:9621",
            "vehiclemind_demo": "http://127.0.0.1:9622",
            "other_car": "http://127.0.0.1:9623",
        },
    )

    with pytest.raises(ValueError, match="instances"):
        ProfileRouter(config)


@pytest.mark.parametrize(
    "url", ["https://evil.example", "http://0.0.0.0:9621", "http://localhost:9621/path"]
)
def test_router_rejects_non_fixed_loopback_url(tmp_path: Path, url: str) -> None:
    config = _config(
        tmp_path,
        instances={
            "vehicle_common": url,
            "vehiclemind_demo": "http://127.0.0.1:9622",
        },
    )

    with pytest.raises(ValueError, match="vehicle_common"):
        ProfileRouter(config)


@pytest.mark.parametrize("timeout", [0, -1, "12"])
def test_router_rejects_non_positive_timeout(tmp_path: Path, timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ProfileRouter(_config(tmp_path, timeout_seconds=timeout))


def test_router_rejects_shared_endpoint(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        instances={
            "vehicle_common": "http://127.0.0.1:9621",
            "vehiclemind_demo": "http://127.0.0.1:9621",
        },
    )

    with pytest.raises(ValueError, match="distinct"):
        ProfileRouter(config)


def test_default_query_timeout_covers_observed_online_retrieval() -> None:
    assert ProfileRouter().timeout_seconds >= 90

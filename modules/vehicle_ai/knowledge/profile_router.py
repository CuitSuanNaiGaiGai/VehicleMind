"""Select a fixed local LightRAG instance for the active profile."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import yaml

_PROFILES = frozenset({"vehicle_common", "vehiclemind_demo"})
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "knowledge.yaml"


class ProfileRouter:
    def __init__(self, config_path: Path = _DEFAULT_CONFIG) -> None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("knowledge config must be a mapping")
        instances = raw.get("instances")
        if not isinstance(instances, dict) or set(instances) != _PROFILES:
            raise ValueError("instances must define exactly two known profiles")
        for profile, url in instances.items():
            if not self._valid_endpoint(url):
                raise ValueError(f"invalid endpoint for {profile}")
        if len(set(instances.values())) != 2:
            raise ValueError("instances must use distinct endpoints")
        timeout = raw.get("timeout_seconds")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        top_k = raw.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")
        self.instances: dict[str, str] = dict(instances)
        self.timeout_seconds: float = float(timeout)
        self.top_k: int = top_k

    def resolve(self, profile: str | None) -> tuple[str, str]:
        selected = profile if profile in _PROFILES else "vehicle_common"
        return selected, self.instances[selected]

    @staticmethod
    def _valid_endpoint(value: object) -> bool:
        if not isinstance(value, str):
            return False
        try:
            url = urlsplit(value)
            return (
                url.scheme == "http"
                and url.hostname == "127.0.0.1"
                and url.port is not None
                and 1 <= url.port <= 65535
                and not url.username
                and not url.password
                and not url.path
                and not url.query
                and not url.fragment
                and value == f"http://127.0.0.1:{url.port}"
            )
        except ValueError:
            return False

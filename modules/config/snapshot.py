from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from modules.config.cabin import CabinPerceptionConfig
from modules.config.perception import PerceptionConfig


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    created_at_utc: str
    git_commit: str
    dirty: bool

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not self.created_at_utc.strip():
            raise ValueError("created_at_utc must be a non-empty string")
        if len(self.git_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.git_commit.lower()
        ):
            raise ValueError("git_commit must be a 40-character hexadecimal commit")


@dataclass(frozen=True)
class RunArtifactPaths:
    manifest: Path
    run_card: Path


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_yaml(value: Mapping[str, object]) -> str:
    return yaml.safe_dump(_plain(value), sort_keys=True, allow_unicode=True)


def build_run_snapshot(
    *,
    cabin: CabinPerceptionConfig,
    perception: PerceptionConfig,
    overrides: Mapping[str, object],
    assets: Sequence[Mapping[str, object]],
    provenance: RunProvenance,
    allow_dirty: bool = False,
) -> dict[str, object]:
    if provenance.dirty and not allow_dirty:
        raise ValueError(
            "formal run refused because of a dirty worktree; "
            "use allow_dirty only for debugging"
        )

    configuration = {
        "cabin": _plain(asdict(cabin)),
        "perception": _plain(asdict(perception)),
        "overrides": _plain(dict(overrides)),
    }
    config_sha256 = hashlib.sha256(
        _canonical_yaml(configuration).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "provenance": _plain(asdict(provenance)),
        "config_sha256": config_sha256,
        "resolved_config": {
            "cabin": configuration["cabin"],
            "perception": configuration["perception"],
        },
        "overrides": configuration["overrides"],
        "model_assets": [_plain(dict(asset)) for asset in assets],
    }


def select_asset_records(
    manifest_path: Path,
    asset_ids: Sequence[str],
) -> list[dict[str, object]]:
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("assets"), list):
        raise ValueError("model asset manifest must contain an assets list")

    assets_by_id = {
        str(asset["id"]): asset
        for asset in document["assets"]
        if isinstance(asset, dict) and "id" in asset
    }
    records: list[dict[str, object]] = []
    for asset_id in asset_ids:
        if asset_id not in assets_by_id:
            raise ValueError(f"unknown model asset id: {asset_id}")
        asset = assets_by_id[asset_id]
        records.append(
            {
                "id": str(asset["id"]),
                "expected_path": str(asset["expected_path"]),
                "size_bytes": int(asset["size_bytes"]),
                "sha256": str(asset["sha256"]),
            }
        )
    return records


def _render_run_card(snapshot: Mapping[str, object]) -> str:
    provenance = snapshot["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("snapshot provenance must be a mapping")
    assets = snapshot.get("model_assets", [])
    if not isinstance(assets, list):
        raise ValueError("snapshot model_assets must be a list")

    lines = [
        "# VehicleMind Run Card",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Run ID | {provenance['run_id']} |",
        f"| Git commit | {provenance['git_commit']} |",
        f"| Dirty worktree | {'yes' if provenance['dirty'] else 'no'} |",
        f"| Config SHA-256 | {snapshot['config_sha256']} |",
        "",
        "## Selected model assets",
        "",
    ]
    if assets:
        lines.extend(
            [
                "| Asset | SHA-256 | Expected path |",
                "|---|---|---|",
            ]
        )
        for asset in assets:
            if not isinstance(asset, Mapping):
                raise ValueError("each model asset must be a mapping")
            lines.append(
                f"| {asset['id']} | {asset['sha256']} | {asset['expected_path']} |"
            )
    else:
        lines.append("_No model assets selected._")

    lines.extend(
        [
            "",
            "## Resolved configuration",
            "",
            "- Cabin configuration: recorded in `resolved_config.yaml`",
            "- Perception configuration: recorded in `resolved_config.yaml`",
            "- CLI overrides: recorded in `resolved_config.yaml`",
            "",
            "## Result status",
            "",
            "No benchmark metrics are recorded in this run artifact.",
            "Accuracy, latency, and qualitative examples must come from a verified "
            "evaluation pipeline.",
            "",
        ]
    )
    return "\n".join(lines)


def write_run_artifacts(
    output_dir: Path,
    snapshot: Mapping[str, object],
) -> RunArtifactPaths:
    manifest = output_dir / "resolved_config.yaml"
    run_card = output_dir / "run_card.md"
    for path in (manifest, run_card):
        if path.exists():
            raise FileExistsError(f"run artifact already exists: {path}")

    manifest_text = yaml.safe_dump(
        _plain(snapshot),
        sort_keys=False,
        allow_unicode=True,
    )
    run_card_text = _render_run_card(snapshot)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_temp = output_dir / ".resolved_config.yaml.tmp"
    run_card_temp = output_dir / ".run_card.md.tmp"
    try:
        manifest_temp.write_text(manifest_text, encoding="utf-8")
        run_card_temp.write_text(run_card_text, encoding="utf-8")
        manifest_temp.replace(manifest)
        run_card_temp.replace(run_card)
    finally:
        manifest_temp.unlink(missing_ok=True)
        run_card_temp.unlink(missing_ok=True)

    return RunArtifactPaths(manifest=manifest, run_card=run_card)

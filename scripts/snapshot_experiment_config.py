from __future__ import annotations

import argparse
import subprocess

from datetime import datetime, timezone
from pathlib import Path

import yaml

from modules.config import CabinPerceptionConfig, PerceptionConfig
from modules.config.overrides import resolve_overrides
from modules.config.snapshot import (
    RunProvenance,
    build_run_snapshot,
    select_asset_records,
    write_run_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_overrides(values: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for value in values:
        path, separator, raw_value = value.partition("=")
        if not separator or not path:
            raise ValueError(f"override must use PATH=VALUE syntax: {value}")
        overrides[path] = yaml.safe_load(raw_value)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible VehicleMind run configuration artifacts"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = _parse_overrides(args.override)
    perception = resolve_overrides(
        PerceptionConfig.load_default(),
        overrides,
    )
    provenance = RunProvenance(
        run_id=args.run_id,
        created_at_utc=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        git_commit=_git("rev-parse", "HEAD"),
        dirty=bool(_git("status", "--porcelain")),
    )
    assets = select_asset_records(
        REPOSITORY_ROOT / "assets/model_manifest.yaml",
        args.asset_id,
    )
    snapshot = build_run_snapshot(
        cabin=CabinPerceptionConfig.load_default(),
        perception=perception,
        overrides=overrides,
        assets=assets,
        provenance=provenance,
        allow_dirty=args.allow_dirty,
    )
    paths = write_run_artifacts(args.output_root / args.run_id, snapshot)
    print(f"Created manifest: {paths.manifest}")
    print(f"Created run card: {paths.run_card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

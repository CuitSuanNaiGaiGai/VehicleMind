from __future__ import annotations

import argparse
import hashlib
import sys

from pathlib import Path
from typing import Any

import yaml


REQUIRED_ASSET_FIELDS = {
    "id",
    "expected_path",
    "size_bytes",
    "sha256",
    "source_url",
    "license",
}
REQUIRED_LICENSE_FIELDS = {"name", "url", "status"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], [f"manifest not found: {path}"]
    except yaml.YAMLError as exc:
        return [], [f"invalid manifest YAML: {exc}"]

    if not isinstance(document, dict) or document.get("version") != 1:
        return [], ["manifest must be a version 1 mapping"]
    assets = document.get("assets")
    if not isinstance(assets, list):
        return [], ["manifest assets must be a list"]

    errors: list[str] = []
    validated: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        label = f"asset[{index}]"
        if not isinstance(asset, dict):
            errors.append(f"{label} must be a mapping")
            continue
        asset_id = str(asset.get("id", label))
        missing = REQUIRED_ASSET_FIELDS - set(asset)
        if missing:
            errors.append(
                f"{asset_id}: missing manifest fields {', '.join(sorted(missing))}"
            )
            continue
        license_data = asset.get("license")
        if not isinstance(license_data, dict):
            errors.append(f"{asset_id}: license must be a mapping")
            continue
        missing_license = REQUIRED_LICENSE_FIELDS - set(license_data)
        if missing_license:
            errors.append(
                f"{asset_id}: missing license fields "
                f"{', '.join(sorted(missing_license))}"
            )
            continue
        validated.append(asset)
    return validated, errors


def verify_assets(manifest_path: Path, root: Path) -> list[str]:
    assets, errors = _load_manifest(manifest_path)
    for asset in assets:
        asset_id = str(asset["id"])
        relative_path = Path(str(asset["expected_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"{asset_id}: expected path must be repository-relative")
            continue

        path = root / relative_path
        required = bool(asset.get("required", True))
        if not path.is_file():
            if required:
                errors.append(
                    f"{asset_id}: expected path {relative_path.as_posix()} is missing"
                )
            continue

        expected_size = asset.get("size_bytes")
        if not isinstance(expected_size, int) or expected_size < 0:
            errors.append(f"{asset_id}: size_bytes must be a non-negative integer")
        elif path.stat().st_size != expected_size:
            errors.append(
                f"{asset_id}: size mismatch at expected path "
                f"{relative_path.as_posix()} (expected {expected_size}, "
                f"got {path.stat().st_size})"
            )

        expected_hash = asset.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            errors.append(f"{asset_id}: sha256 must contain 64 hexadecimal characters")
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash.lower():
            errors.append(
                f"{asset_id}: SHA-256 mismatch at expected path "
                f"{relative_path.as_posix()} (expected {expected_hash.lower()}, "
                f"got {actual_hash})"
            )
    return errors


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify VehicleMind model assets")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "assets/model_manifest.yaml",
    )
    parser.add_argument("--root", type=Path, default=repository_root)
    args = parser.parse_args()

    errors = verify_assets(args.manifest, args.root)
    if errors:
        print("Asset verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("All required model assets match the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import hashlib
import re
import sys

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _is_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_asset_values(
    asset: dict[str, Any],
    asset_id: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(asset.get("id"), str) or not asset["id"].strip():
        errors.append(f"{asset_id}: id must be a non-empty string")

    expected_path = asset.get("expected_path")
    if not isinstance(expected_path, str) or not expected_path:
        errors.append(f"{asset_id}: expected path must be a non-empty string")
    else:
        relative_path = Path(expected_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"{asset_id}: expected path must be repository-relative")

    size_bytes = asset.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        errors.append(f"{asset_id}: size_bytes must be a non-negative integer")

    sha256 = asset.get("sha256")
    if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
        errors.append(f"{asset_id}: sha256 must contain 64 hexadecimal characters")

    if not _is_http_url(asset.get("source_url")):
        errors.append(f"{asset_id}: source_url must be an absolute HTTP(S) URL")

    required = asset.get("required", True)
    if not isinstance(required, bool):
        errors.append(f"{asset_id}: required must be a boolean")

    license_data = asset.get("license")
    if isinstance(license_data, dict):
        for field in ("name", "status"):
            value = license_data.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{asset_id}: license {field} must be a non-empty string")
        if not _is_http_url(license_data.get("url")):
            errors.append(f"{asset_id}: license url must be an absolute HTTP(S) URL")
    return errors


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
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
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
        value_errors = _validate_asset_values(asset, asset_id)
        errors.extend(value_errors)

        if isinstance(asset.get("id"), str):
            if asset["id"] in seen_ids:
                errors.append(f"{asset_id}: duplicate asset id")
            seen_ids.add(asset["id"])
        if isinstance(asset.get("expected_path"), str):
            if asset["expected_path"] in seen_paths:
                errors.append(f"{asset_id}: duplicate expected path")
            seen_paths.add(asset["expected_path"])

        if not value_errors:
            validated.append(asset)
    return validated, errors


def verify_assets(manifest_path: Path, root: Path) -> list[str]:
    assets, errors = _load_manifest(manifest_path)
    for asset in assets:
        asset_id = str(asset["id"])
        relative_path = Path(str(asset["expected_path"]))
        path = root / relative_path
        required = bool(asset.get("required", True))
        if not path.is_file():
            if required:
                errors.append(
                    f"{asset_id}: expected path {relative_path.as_posix()} is missing"
                )
            continue

        expected_size = int(asset["size_bytes"])
        if path.stat().st_size != expected_size:
            errors.append(
                f"{asset_id}: size mismatch at expected path "
                f"{relative_path.as_posix()} (expected {expected_size}, "
                f"got {path.stat().st_size})"
            )

        expected_hash = str(asset["sha256"])
        actual_hash = _sha256(path)
        if actual_hash != expected_hash.lower():
            errors.append(
                f"{asset_id}: SHA-256 mismatch at expected path "
                f"{relative_path.as_posix()} (expected {expected_hash.lower()}, "
                f"got {actual_hash})"
            )
    return errors


def validate_manifest(manifest_path: Path) -> list[str]:
    _, errors = _load_manifest(manifest_path)
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
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate manifest structure without requiring local model files",
    )
    args = parser.parse_args()

    errors = (
        validate_manifest(args.manifest)
        if args.schema_only
        else verify_assets(args.manifest, args.root)
    )
    if errors:
        print("Asset verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("All required model assets match the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

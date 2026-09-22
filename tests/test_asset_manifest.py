from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.verify_assets import validate_manifest, verify_assets


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(path: Path, assets: list[dict[str, object]]) -> None:
    path.write_text(
        yaml.safe_dump({"version": 1, "assets": assets}, sort_keys=False),
        encoding="utf-8",
    )


def _valid_asset() -> dict[str, object]:
    return {
        "id": "test_asset",
        "expected_path": "models/test.bin",
        "size_bytes": 10,
        "sha256": "0" * 64,
        "source_url": "https://example.invalid/test.bin",
        "license": {
            "name": "test-only",
            "url": "https://example.invalid/license",
            "status": "test-only",
        },
    }


def test_repository_manifest_schema_is_valid() -> None:
    assert validate_manifest(REPOSITORY_ROOT / "assets/model_manifest.yaml") == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("size_bytes", -1, "non-negative integer"),
        ("sha256", "z" * 64, "64 hexadecimal"),
        ("expected_path", "/tmp/model.bin", "repository-relative"),
        ("source_url", "not-a-url", "absolute HTTP(S) URL"),
    ),
)
def test_schema_rejects_invalid_asset_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    asset = _valid_asset()
    asset[field] = value
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, [asset])

    errors = validate_manifest(manifest_path)

    assert any(message in error for error in errors)


def test_schema_rejects_duplicate_ids_and_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, [_valid_asset(), _valid_asset()])

    errors = validate_manifest(manifest_path)

    assert any("duplicate asset id" in error for error in errors)
    assert any("duplicate expected path" in error for error in errors)


def test_missing_repository_assets_have_actionable_errors(tmp_path: Path) -> None:
    errors = verify_assets(
        REPOSITORY_ROOT / "assets/model_manifest.yaml",
        tmp_path,
    )

    assert len(errors) == 4
    assert all("expected path" in error for error in errors)


def test_matching_asset_passes_verification(tmp_path: Path) -> None:
    content = b"vehiclemind-test-asset"
    asset_path = tmp_path / "models/test.bin"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(content)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "assets": [
                    {
                        "id": "test_asset",
                        "expected_path": "models/test.bin",
                        "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "source_url": "https://example.invalid/test.bin",
                        "license": {
                            "name": "test-only",
                            "url": "https://example.invalid/license",
                            "status": "test-only",
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert verify_assets(manifest_path, tmp_path) == []


def test_hash_mismatch_is_reported(tmp_path: Path) -> None:
    content = b"unexpected"
    asset_path = tmp_path / "models/test.bin"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(content)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "version: 1\n"
        "assets:\n"
        "  - id: test_asset\n"
        "    expected_path: models/test.bin\n"
        f"    size_bytes: {len(content)}\n"
        f'    sha256: "{"0" * 64}"\n'
        "    source_url: https://example.invalid/test.bin\n"
        "    license:\n"
        "      name: test-only\n"
        "      url: https://example.invalid/license\n"
        "      status: test-only\n",
        encoding="utf-8",
    )

    errors = verify_assets(manifest_path, tmp_path)

    assert errors == [
        "test_asset: SHA-256 mismatch at expected path models/test.bin "
        f"(expected {'0' * 64}, got {hashlib.sha256(content).hexdigest()})"
    ]

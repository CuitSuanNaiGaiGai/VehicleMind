from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from scripts.verify_assets import verify_assets


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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

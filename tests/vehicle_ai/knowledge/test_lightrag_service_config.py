"""The two LightRAG processes must never share storage or scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.lightrag_services import (
    build_command,
    build_provider_env,
    build_service_specs,
    validate_sidecar_config,
)


def test_service_specs_use_distinct_fixed_directories_and_ports(tmp_path: Path) -> None:
    specs = build_service_specs(tmp_path)

    assert tuple(specs) == ("vehicle_common", "vehiclemind_demo")
    assert specs["vehicle_common"].port == 9621
    assert specs["vehiclemind_demo"].port == 9622
    assert specs["vehicle_common"].work_dir != specs["vehiclemind_demo"].work_dir
    assert specs["vehicle_common"].input_dir != specs["vehiclemind_demo"].input_dir
    assert specs["vehicle_common"].work_dir == (
        tmp_path / "runs/lightrag/vehicle_common/storage"
    )
    assert specs["vehiclemind_demo"].work_dir == (
        tmp_path / "runs/lightrag/vehiclemind_demo/storage"
    )
    validate_sidecar_config(specs, tmp_path)


def test_config_rejects_index_directory_swapped_between_profiles(
    tmp_path: Path,
) -> None:
    specs = build_service_specs(tmp_path)
    common = specs["vehicle_common"]
    specs["vehicle_common"] = common._replace(
        work_dir=specs["vehiclemind_demo"].work_dir
    )

    with pytest.raises(ValueError, match="vehicle_common"):
        validate_sidecar_config(specs, tmp_path)


def test_config_rejects_port_mismatch_with_router(tmp_path: Path) -> None:
    specs = build_service_specs(tmp_path)
    specs["vehicle_common"] = specs["vehicle_common"]._replace(port=9623)

    with pytest.raises(ValueError, match="vehicle_common"):
        validate_sidecar_config(specs, tmp_path)


def test_command_binds_loopback_and_profile_storage(tmp_path: Path) -> None:
    spec = build_service_specs(tmp_path)["vehicle_common"]

    assert build_command(
        spec, tmp_path / "tools/lightrag/.venv/bin/lightrag-server"
    ) == [
        str(tmp_path / "tools/lightrag/.venv/bin/lightrag-server"),
        "--host",
        "127.0.0.1",
        "--port",
        "9621",
        "--working-dir",
        str(tmp_path / "runs/lightrag/vehicle_common/storage"),
        "--input-dir",
        str(tmp_path / "runs/lightrag/vehicle_common/inputs"),
    ]


def test_provider_env_maps_existing_qwen_settings_without_logging_secrets() -> None:
    source = {
        "DASHSCOPE_API_KEY": "private-test-key",
        "QWEN_MODEL": "qwen3.8-max",
        "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }

    result = build_provider_env(source)

    assert result["LLM_BINDING"] == "openai"
    assert result["LLM_MODEL"] == "qwen3.8-max"
    assert result["LLM_BINDING_API_KEY"] == "private-test-key"
    assert result["EMBEDDING_BINDING"] == "openai"
    assert result["EMBEDDING_MODEL"] == "text-embedding-v3"
    assert result["EMBEDDING_DIM"] == "1024"
    assert result["EMBEDDING_BINDING_API_KEY"] == "private-test-key"
    assert result["EMBEDDING_BINDING_HOST"] == source["QWEN_BASE_URL"]
    assert "DASHSCOPE_API_KEY" not in result


@pytest.mark.parametrize(
    "missing", ["DASHSCOPE_API_KEY", "QWEN_MODEL", "QWEN_BASE_URL"]
)
def test_provider_env_requires_each_existing_setting(missing: str) -> None:
    source = {
        "DASHSCOPE_API_KEY": "private-test-key",
        "QWEN_MODEL": "qwen3.8-max",
        "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
    del source[missing]

    with pytest.raises(ValueError, match=missing):
        build_provider_env(source)

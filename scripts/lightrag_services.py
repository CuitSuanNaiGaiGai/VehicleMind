"""Run two fixed local LightRAG services with separate persistent stores."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import dotenv_values

from modules.vehicle_ai.knowledge.profile_router import ProfileRouter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("vehicle_common", "vehiclemind_demo")


class ServiceSpec(NamedTuple):
    profile: str
    port: int
    work_dir: Path
    input_dir: Path


def build_service_specs(root: Path = REPOSITORY_ROOT) -> dict[str, ServiceSpec]:
    router = ProfileRouter()
    specs: dict[str, ServiceSpec] = {}
    for profile in PROFILES:
        port = int(router.instances[profile].rsplit(":", 1)[1])
        instance_root = root / "runs" / "lightrag" / profile
        specs[profile] = ServiceSpec(
            profile=profile,
            port=port,
            work_dir=instance_root / "storage",
            input_dir=instance_root / "inputs",
        )
    validate_sidecar_config(specs, root)
    return specs


def validate_sidecar_config(
    specs: Mapping[str, ServiceSpec], root: Path = REPOSITORY_ROOT
) -> None:
    router = ProfileRouter()
    if tuple(specs) != PROFILES:
        raise ValueError("sidecar instances must use the two fixed profiles")
    if len({spec.port for spec in specs.values()}) != 2:
        raise ValueError("sidecar instances need distinct ports")
    for profile in PROFILES:
        spec = specs[profile]
        expected = (root / "runs" / "lightrag" / profile).resolve()
        if (
            spec.profile != profile
            or spec.port != int(router.instances[profile].rsplit(":", 1)[1])
            or spec.work_dir.resolve() != expected / "storage"
            or spec.input_dir.resolve() != expected / "inputs"
        ):
            raise ValueError(f"{profile} uses a mismatched index directory")


def build_command(spec: ServiceSpec, executable: Path) -> list[str]:
    return [
        str(executable),
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--working-dir",
        str(spec.work_dir),
        "--input-dir",
        str(spec.input_dir),
    ]


def build_provider_env(source: Mapping[str, str]) -> dict[str, str]:
    for key in ("DASHSCOPE_API_KEY", "QWEN_MODEL", "QWEN_BASE_URL"):
        if not source.get(key):
            raise ValueError(f"missing {key}")
    return {
        "LLM_BINDING": "openai",
        "LLM_MODEL": source["QWEN_MODEL"],
        "LLM_BINDING_HOST": source["QWEN_BASE_URL"],
        "LLM_BINDING_API_KEY": source["DASHSCOPE_API_KEY"],
        "EMBEDDING_BINDING": "openai",
        "EMBEDDING_MODEL": "text-embedding-v3",
        "EMBEDDING_BINDING_HOST": source["QWEN_BASE_URL"],
        "EMBEDDING_BINDING_API_KEY": source["DASHSCOPE_API_KEY"],
        "EMBEDDING_DIM": "1024",
        "EMBEDDING_SEND_DIM": "false",
        "EMBEDDING_USE_BASE64": "false",
        "EMBEDDING_ASYMMETRIC": "false",
        "RERANK_BINDING": "null",
    }


def _provider_source(root: Path) -> dict[str, str]:
    file_values = dotenv_values(root / ".env")
    values = {key: value for key, value in file_values.items() if value is not None}
    values.update(os.environ)
    return values


def healthcheck_services(
    specs: Mapping[str, ServiceSpec], *, timeout_seconds: float = 2.0
) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for profile, spec in specs.items():
        try:
            with urlopen(
                f"http://127.0.0.1:{spec.port}/health", timeout=timeout_seconds
            ) as response:
                status[profile] = response.status == 200
        except (OSError, URLError):
            status[profile] = False
    return status


def start_services(
    specs: Mapping[str, ServiceSpec],
    *,
    root: Path = REPOSITORY_ROOT,
    executable: Path | None = None,
) -> dict[str, subprocess.Popen[bytes]]:
    validate_sidecar_config(specs, root)
    binary = executable or root / "tools/lightrag/.venv/bin/lightrag-server"
    if not binary.is_file():
        raise FileNotFoundError(f"LightRAG server executable missing: {binary}")
    environment = {**os.environ, **build_provider_env(_provider_source(root))}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        for profile, spec in specs.items():
            spec.work_dir.mkdir(parents=True, exist_ok=True)
            spec.input_dir.mkdir(parents=True, exist_ok=True)
            log_path = spec.work_dir.parent / "server.log"
            with log_path.open("ab") as log:
                processes[profile] = subprocess.Popen(
                    build_command(spec, binary),
                    cwd=spec.work_dir.parent,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        return processes
    except Exception:
        stop_services(processes)
        raise


def stop_services(processes: Mapping[str, subprocess.Popen[bytes]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _run() -> int:
    parser = argparse.ArgumentParser(
        description="运行两个固定 profile 的 LightRAG 服务"
    )
    parser.add_argument("command", choices=("run", "health"))
    args = parser.parse_args()
    specs = build_service_specs()
    if args.command == "health":
        status = healthcheck_services(specs)
        for profile, healthy in status.items():
            print(f"{profile}: {'ready' if healthy else 'unavailable'}")
        return 0 if all(status.values()) else 1
    processes = start_services(specs)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = healthcheck_services(specs)
            if all(status.values()):
                print("LightRAG services ready: vehicle_common, vehiclemind_demo")
                break
            if any(process.poll() is not None for process in processes.values()):
                raise RuntimeError("a LightRAG service exited during startup")
            time.sleep(1)
        else:
            raise TimeoutError("LightRAG service startup timed out")
        while all(process.poll() is None for process in processes.values()):
            time.sleep(1)
        raise RuntimeError("a LightRAG service exited")
    except KeyboardInterrupt:
        return 0
    finally:
        stop_services(processes)


if __name__ == "__main__":
    raise SystemExit(_run())

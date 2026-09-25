"""Build LightRAG indexes in temporary services, then publish complete stores."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog
from modules.vehicle_ai.knowledge.index_builder import (
    IndexBuilder,
    LightRAGDocumentClient,
    publish_staged_index,
)
from scripts.lightrag_services import (
    REPOSITORY_ROOT,
    ServiceSpec,
    _provider_source,
    build_command,
    build_provider_env,
    build_service_specs,
    healthcheck_services,
    stop_services,
)


def _wait_for_service(spec: ServiceSpec, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("staging LightRAG service exited during startup")
        if all(healthcheck_services({spec.profile: spec}).values()):
            return
        time.sleep(1)
    raise TimeoutError("staging LightRAG service startup timed out")


def build_profile_index(profile: str, root: Path = REPOSITORY_ROOT) -> None:
    specs = build_service_specs(root)
    if profile not in specs:
        raise ValueError(f"unknown profile: {profile}")
    active_spec = specs[profile]
    if healthcheck_services({profile: active_spec})[profile]:
        raise RuntimeError(f"stop active {profile} service before rebuilding")
    sources = KnowledgeCatalog().load(root / "config/knowledge/source_catalog.yaml")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    stage_root = root / "runs/lightrag/.staging" / f"{profile}-{run_id}"
    stage_spec = active_spec._replace(
        work_dir=stage_root / "storage", input_dir=stage_root / "inputs"
    )
    stage_spec.work_dir.mkdir(parents=True)
    stage_spec.input_dir.mkdir()
    binary = root / "tools/lightrag/.venv/bin/lightrag-server"
    if not binary.is_file():
        raise FileNotFoundError(f"LightRAG server executable missing: {binary}")
    environment = {**os.environ, **build_provider_env(_provider_source(root))}
    log_path = stage_root / "server.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            build_command(stage_spec, binary),
            cwd=stage_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        _wait_for_service(stage_spec, process)
        print(f"{profile}: 暂存服务就绪，开始索引来源文档", flush=True)
        client = LightRAGDocumentClient(
            f"http://127.0.0.1:{stage_spec.port}", timeout_seconds=90
        )
        result = IndexBuilder(
            client, stage_spec.work_dir, poll_seconds=5, timeout_seconds=3600
        ).build(profile, sources)
        print(
            f"{profile}: {result.source_count} 条处理完成，"
            f"manifest_sha256={result.manifest_sha256}",
            flush=True,
        )
    finally:
        stop_services({profile: process})
    active_root = root / "runs/lightrag" / profile
    backup_root = root / "runs/lightrag/.backups" / f"{profile}-{run_id}"
    publish_staged_index(stage_root, active_root, backup_root)
    print(f"{profile}: 已发布完整索引；旧索引备份位置 {backup_root}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="暂存构建并发布 A3 知识索引")
    parser.add_argument(
        "--profile",
        choices=("all", "vehicle_common", "vehiclemind_demo"),
        default="all",
    )
    args = parser.parse_args()
    profiles = (
        ("vehicle_common", "vehiclemind_demo")
        if args.profile == "all"
        else (args.profile,)
    )
    for profile in profiles:
        build_profile_index(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

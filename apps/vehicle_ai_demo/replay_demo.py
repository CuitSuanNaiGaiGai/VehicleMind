from __future__ import annotations

import argparse
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path

from modules.config import CabinPerceptionConfig, PerceptionConfig
from modules.config.events import EventTimingConfig
from modules.config.snapshot import RunProvenance, build_run_snapshot
from modules.vehicle_ai.replay import ReplayRunner, load_replay_scenario
from modules.vehicle_ai.replay.report import write_replay_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_scenario(path: Path) -> tuple[Path, str]:
    resolved = (REPOSITORY_ROOT / path).resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT):
        relative = resolved.relative_to(REPOSITORY_ROOT)
        return resolved, relative.as_posix()
    return resolved, "external-scenario"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行可复现的 VehicleMind 离线回放场景"
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    scenario_path, scenario_reference = _resolve_scenario(args.scenario)
    scenario = load_replay_scenario(
        scenario_path,
        repository_root=REPOSITORY_ROOT,
    )
    run_id = args.run_id or scenario.scenario_id
    provenance = RunProvenance(
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        git_commit=_git("rev-parse", "HEAD"),
        dirty=bool(_git("status", "--porcelain")),
    )
    event_timing = EventTimingConfig.load()
    snapshot = build_run_snapshot(
        cabin=CabinPerceptionConfig.load_default(),
        perception=PerceptionConfig.load_default(),
        events=event_timing,
        overrides={"scenario": scenario_reference},
        assets=[],
        provenance=provenance,
        allow_dirty=args.allow_dirty,
    )
    result = ReplayRunner(event_timing=event_timing).run(scenario)
    paths = write_replay_report(
        result,
        scenario,
        snapshot,
        args.output_root / run_id,
    )

    status = "通过" if result.passed else "失败"
    print(f"{status}: {scenario.scenario_id}")
    print(f"语义追踪 SHA-256：{result.semantic_sha256}")
    print(f"报告页面：{paths.report_html.resolve()}")
    print(f"结果摘要：{paths.summary_json.resolve()}")
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.debug:
        return _run(args)
    try:
        return _run(args)
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

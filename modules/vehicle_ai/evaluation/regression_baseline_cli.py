"""Prepare an immutable historical Agent run for current-protocol re-review."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from modules.vehicle_ai.evaluation.batch import load_frozen_cases
from modules.vehicle_ai.evaluation.regression_baseline import prepare_rejudge_copy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="准备历史 Agent run 的隔离复核副本")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--golden", type=Path, default=Path("scenarios/agent_eval/golden")
    )
    parser.add_argument("--source-revision", default="c0a314e")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("重复次数必须为正数")
    repo_root = Path(__file__).resolve().parents[3]
    grading_revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cases = load_frozen_cases(args.golden)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = prepare_rejudge_copy(
        args.source,
        args.output,
        dataset_root=args.golden,
        expected_case_ids=tuple(case.id for case in cases),
        repetitions=args.repetitions,
        source_revision=args.source_revision,
        grading_revision=grading_revision,
    )
    print(
        f"已生成 {result['trial_count']} 条隔离复核 trace：{args.output.name}；"
        f"原始 run 未修改。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

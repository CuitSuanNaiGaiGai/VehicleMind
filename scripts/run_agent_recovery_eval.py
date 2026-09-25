from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from modules.vehicle_ai.evaluation.recovery import run_recovery_suite
from modules.vehicle_ai.evaluation.recovery_report import write_recovery_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 VehicleMind Agent 受限计划与恢复的确定性验收。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="结果目录，默认写入带 UTC 时间戳的新目录。",
    )
    args = parser.parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("runs/agent_recovery") / run_id
    report = run_recovery_suite()
    paths = write_recovery_report(report, output)
    print(
        f"A5 场景通过：{sum(case.passed for case in report.cases)}/{len(report.cases)}"
    )
    print(f"中文报告：{paths['html']}")
    print(f"结构化结果：{paths['json']}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

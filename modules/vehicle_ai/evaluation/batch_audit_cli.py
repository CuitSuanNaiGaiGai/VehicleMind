"""Apply documented corrections to AI-assisted Agent batch judgments."""

from __future__ import annotations

import argparse
from pathlib import Path

from modules.vehicle_ai.evaluation.batch_audit import audit_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 内部语义审核纠错")
    parser.add_argument("run", type=Path)
    parser.add_argument("overrides", type=Path)
    args = parser.parse_args(argv)
    result = audit_batch(args.run, args.overrides)
    print(f"复查后 Task Success：{result['task_success']}")
    print(f"已保留原始审核，修正记录见 {args.run / 'audited.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

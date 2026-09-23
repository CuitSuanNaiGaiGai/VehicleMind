"""Export model-anonymous packets and validate human review decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.review import build_blind_packet, validate_decision
from modules.vehicle_ai.evaluation.rubric import load_rubric
from modules.vehicle_ai.evaluation.runner import TrialResult


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_new(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 事实依据盲审")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="从 trace 导出隐藏模型身份的审查包")
    export.add_argument("--case", type=Path, required=True)
    export.add_argument("--rubric", type=Path, required=True)
    export.add_argument("--trial", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate", help="校验人工审查决定")
    validate.add_argument("--packet", type=Path, required=True)
    validate.add_argument("--decision", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "export":
        case = EvaluationCase.from_mapping(
            yaml.safe_load(args.case.read_text(encoding="utf-8"))
        )
        rubric = load_rubric(args.rubric, case)
        trace = _json(args.trial)
        trial = TrialResult(**trace["trial"])
        packet = build_blind_packet(case, rubric, trial)
        _write_new(args.output, packet)
        print(f"盲审材料：{args.output}")
        return 0

    packet = _json(args.packet)
    decision = validate_decision(packet, _json(args.decision))
    _write_new(args.output, decision)
    print(f"审查决定：{args.output}")
    print(f"正式成绩资格：{'是' if decision['formal_eligible'] else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

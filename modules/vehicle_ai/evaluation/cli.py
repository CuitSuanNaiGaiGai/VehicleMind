from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.report import write_report
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm import build_llm_client


def main() -> int:
    parser = argparse.ArgumentParser(description="在线 Agent 候选场景评估")
    parser.add_argument("case", type=Path, help="结构化 YAML 场景")
    parser.add_argument("--provider", choices=("qwen", "glm"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/agent_eval"))
    parser.add_argument("--trial-index", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-tool-rounds", type=int, default=5)
    args = parser.parse_args()
    load_dotenv()
    if args.timeout_seconds <= 0 or args.max_tool_rounds <= 0:
        parser.error("timeout 和最大工具轮次必须为正数")
    document = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    case = EvaluationCase.from_mapping(document)
    default_model = (
        os.getenv("QWEN_MODEL", "qwen3.8-max")
        if args.provider == "qwen"
        else os.getenv("GLM_MODEL", "glm-5.1")
    )
    client = build_llm_client(
        args.provider,
        model=args.model or default_model,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        args.output_dir / f"{stamp}-{args.provider}-{case.id}-t{args.trial_index}"
    )
    trial = run_trial(
        case,
        client,
        provider=args.provider,
        model=str(getattr(client, "model")),
        trial_index=args.trial_index,
        max_tool_rounds=args.max_tool_rounds,
    )
    grade = grade_trial(case, trial)
    write_report(destination, case, trial, grade)
    print(f"评估报告：{destination / 'report.md'}")
    print(f"判定：{grade['status']}（候选数据，不计正式成功率）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry point for the eight-case candidate pilot."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from modules.vehicle_ai.evaluation.pilot import load_pilot_cases, run_pilot
from modules.vehicle_ai.llm import build_llm_client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双模型在线 Agent 候选集 pilot")
    parser.add_argument("--provider", choices=("qwen", "glm"), required=True)
    parser.add_argument(
        "--cases", type=Path,
        default=Path("scenarios/agent_eval/candidates"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/agent_eval"))
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("请求超时必须为正数")
    load_dotenv()
    model = args.model or (
        os.getenv("QWEN_MODEL", "qwen3.8-max") if args.provider == "qwen"
        else os.getenv("GLM_MODEL", "glm-5.1")
    )
    client = build_llm_client(
        args.provider, model=model, temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )
    cases = load_pilot_cases(args.cases)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_root / f"pilot-{stamp}-{args.provider}"
    result = run_pilot(
        args.provider, client, cases, output_root=output, model=model,
    )
    print(f"预检：{'通过' if result['preflight']['passed'] else '失败'}")
    print(f"已运行候选场景：{len(result['cases'])} / {len(cases)}")
    failed_cases = sum(entry["status"] == "fail" for entry in result["cases"])
    errored_cases = sum(entry["error"] is not None for entry in result["cases"])
    print(f"模型评分失败：{failed_cases}；运行异常：{errored_cases}")
    print(f"运行记录：{output / 'pilot.json'}")
    if not result["preflight"]["passed"]:
        return 2
    return 3 if errored_cases or result["status"] != "completed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

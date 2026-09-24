"""Run an explicitly AI-assisted semantic review over one completed batch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from modules.vehicle_ai.evaluation.batch_judge import judge_batch
from modules.vehicle_ai.llm import build_llm_client


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在线 Agent 内部结果 AI 辅助复核")
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--golden", type=Path, default=Path("scenarios/agent_eval/golden")
    )
    parser.add_argument("--provider", choices=("qwen", "glm"), default="qwen")
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    load_dotenv()
    model = args.model or os.getenv(
        "QWEN_MODEL" if args.provider == "qwen" else "GLM_MODEL",
        "qwen3.8-max" if args.provider == "qwen" else "glm-5.1",
    )
    client = build_llm_client(
        args.provider, model=model, temperature=0.0, timeout_seconds=90.0
    )
    result = judge_batch(
        args.run, args.golden, client, judge_model=f"{args.provider}/{model}"
    )
    print(
        f"AI 辅助复核完成：{result['task_success']}; {args.run / 'reviewed_summary.md'}"
    )
    print("此结果不是独立人工评测；审核原文与逐条决定已保留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run repeated online Agent trials against hash-frozen internal cases."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from modules.vehicle_ai.evaluation.batch import load_frozen_cases, run_batch
from modules.vehicle_ai.llm import build_llm_client
from modules.vehicle_ai.agent.budget import AgentBudgetConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在线 Agent 内部重复评测")
    parser.add_argument("--provider", choices=("qwen", "glm"), required=True)
    parser.add_argument(
        "--golden", type=Path, default=Path("scenarios/agent_eval/golden")
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/agent_eval"))
    parser.add_argument(
        "--case-id", action="append", default=[], help="仅跑指定场景；默认全部 40 条"
    )
    parser.add_argument("--split", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-tool-rounds", type=int)
    parser.add_argument(
        "--agent-config", type=Path, default=Path("modules/config/agent.yaml")
    )
    parser.add_argument("--turn-timeout-seconds", type=float)
    parser.add_argument("--max-tool-calls", type=int)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.timeout_seconds <= 0:
        parser.error("重复次数、超时和最大工具轮次必须为正数")
    config = AgentBudgetConfig.load(args.agent_config)
    turn_timeout = args.turn_timeout_seconds or config.turn_timeout_seconds
    tool_calls = args.max_tool_calls or config.max_tool_calls
    tool_rounds = args.max_tool_rounds or config.max_tool_rounds
    if turn_timeout <= 0 or tool_calls <= 0 or tool_rounds <= 0:
        parser.error("Agent 时间预算和工具次数预算必须为正数")
    cases = load_frozen_cases(args.golden)
    if args.split != "all":
        cases = tuple(case for case in cases if case.split == args.split)
    if args.case_id:
        wanted = set(args.case_id)
        available = {case.id for case in cases}
        if wanted - available:
            parser.error(f"未知场景：{', '.join(sorted(wanted - available))}")
        cases = tuple(case for case in cases if case.id in wanted)
    load_dotenv()
    default_model = (
        os.getenv("QWEN_MODEL", "qwen3.8-max")
        if args.provider == "qwen"
        else os.getenv("GLM_MODEL", "glm-5.1")
    )
    model = args.model or default_model
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_root / f"batch-{stamp}-{args.provider}"
    result = run_batch(
        cases,
        provider=args.provider,
        model=model,
        client_factory=lambda: build_llm_client(
            args.provider,
            model=model,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
        ),
        repetitions=args.repetitions,
        output=output,
        max_tool_rounds=tool_rounds,
        turn_timeout_seconds=turn_timeout,
        max_tool_calls=tool_calls,
        max_task_trace_events=config.max_task_trace_events,
    )
    print(f"已保存 {len(result['trials'])} 个 trial：{output}")
    print("回答语义仍待逐条复核；不能把 needs_review 计为成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

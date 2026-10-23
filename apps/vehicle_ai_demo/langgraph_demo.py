"""可跨进程确认的 LangGraph 座舱演示，默认使用明确的脚本模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from modules.vehicle_ai.llm import build_llm_client
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="VehicleMind LangGraph 状态与确认恢复演示"
    )
    parser.add_argument(
        "command", choices=("chat", "confirm", "reject", "inspect", "graph", "recover")
    )
    parser.add_argument(
        "--checkpoint-db", type=Path, default=Path("runs/state/langgraph.sqlite3")
    )
    parser.add_argument("--thread-id", default="interview-demo")
    parser.add_argument("--text", default="帮我找附近服务区并导航")
    parser.add_argument("--action-id", help="确认/拒绝必须显式指定当前动作 ID")
    parser.add_argument(
        "--provider", choices=("scripted", "qwen", "glm"), default="scripted"
    )
    parser.add_argument("--output", type=Path, help="保存图 Mermaid 或状态 JSON")
    return parser.parse_args(argv)


def _scripted_client():
    return ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall("rest-search", "search_nearby_rest_area", {}, "{}"),
                ),
            ),
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        "rest-navigation",
                        "start_navigation",
                        {"poi_id": "rest_area_001"},
                        '{"poi_id":"rest_area_001"}',
                    ),
                ),
            ),
            ScriptedResponse(content="这是脚本模型演示；请求需要重新说明。"),
        )
    )


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    args = parse_args(argv)
    if args.command in {"confirm", "reject"} and not args.action_id:
        raise ValueError("确认或拒绝必须提供 --action-id，不能自动接受任意待确认动作")
    client = (
        _scripted_client()
        if args.provider == "scripted"
        else build_llm_client(args.provider)
    )
    runtime = VehicleMindRuntime(
        client,
        checkpoint_path=args.checkpoint_db,
        thread_id=args.thread_id,
        enable_event_recommendations=False,
    )
    try:
        if args.command == "graph":
            output = runtime.agent.graph.get_graph().draw_mermaid()
        else:
            result = None
            if args.command == "chat":
                result = runtime.chat(args.text, debug=False)
            elif args.command == "confirm":
                result = runtime.agent.confirm_pending(args.action_id).to_dict()
            elif args.command == "reject":
                result = runtime.agent.reject_pending(args.action_id).to_dict()
            elif args.command == "recover":
                result = runtime.agent.workflow.recover().get("final_response")
            snapshot = runtime.agent.graph.get_state(runtime.agent.workflow.config)
            pending = runtime.agent.pending_actions.get()
            output = json.dumps(
                {
                    "provider": args.provider,
                    "thread_id": runtime.agent.thread_id,
                    "result": result,
                    "next": list(snapshot.next),
                    "task": runtime.agent.task.to_dict(),
                    "pending_action": pending.to_dict() if pending else None,
                    "graph_trace": snapshot.values.get("graph_trace", []),
                    "vehicle": runtime.context_manager.get_context().vehicle.__dict__,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        print(output)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

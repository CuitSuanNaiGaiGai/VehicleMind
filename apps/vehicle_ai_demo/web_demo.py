"""运行 VehicleMind 的真实 LangGraph 交互演示。"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from apps.vehicle_ai_demo.web_http import DemoHTTPServer
from apps.vehicle_ai_demo.web_session import SessionRegistry


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    parser = argparse.ArgumentParser(description="VehicleMind LangGraph 浏览器演示")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--checkpoint-db", type=Path, default=Path("runs/state/web-langgraph.sqlite3")
    )
    parser.add_argument(
        "--provider", choices=("scripted", "qwen", "glm"), default="scripted"
    )
    parser.add_argument(
        "--demo-delay",
        type=float,
        default=0.6,
        help="离线脚本模型的演示响应延迟，单位秒",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.demo_delay <= 5:
        raise ValueError("demo-delay 必须在 0 到 5 秒之间")
    registry = SessionRegistry(args.checkpoint_db, args.demo_delay, args.provider)
    server = DemoHTTPServer(("127.0.0.1", args.port), registry)
    print(f"VehicleMind 交互演示：http://127.0.0.1:{server.server_port}", flush=True)
    print(
        f"模型：{args.provider}；checkpoint：{args.checkpoint_db.resolve()}", flush=True
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        registry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

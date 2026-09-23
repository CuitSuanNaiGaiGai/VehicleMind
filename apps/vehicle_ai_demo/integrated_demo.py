from __future__ import annotations

import argparse
import math
import time

from pathlib import Path

from dotenv import load_dotenv

from apps.vehicle_ai_demo.terminal_display import (
    format_context,
    format_event,
    format_freshness,
    format_health,
)
from apps.vehicle_ai_demo.video_pipelines import (
    DEFAULT_CABIN_VIDEO,
    DEFAULT_ROAD_VIDEO,
    DEFAULT_CABIN_MODEL,
    DEFAULT_ROAD_MODEL,
    build_cabin_pipeline,
    build_road_pipeline,
)

from modules.vehicle_ai.context import (
    GearState,
)

from modules.vehicle_ai.llm import (
    build_llm_client,
)

from modules.vehicle_ai.runtime import (
    VehicleMindRuntime,
)
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("数值必须为正")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("数值必须为正")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VehicleMind 离线视频协同演示")
    parser.add_argument("--cabin-video", type=Path, default=DEFAULT_CABIN_VIDEO)
    parser.add_argument("--road-video", type=Path, default=DEFAULT_ROAD_VIDEO)
    parser.add_argument("--cabin-model", type=Path, default=DEFAULT_CABIN_MODEL)
    parser.add_argument("--road-model", type=Path, default=DEFAULT_ROAD_MODEL)
    parser.add_argument("--cabin-hz", type=_positive_float, default=12.0)
    parser.add_argument("--road-hz", type=_positive_float, default=8.0)
    parser.add_argument("--frame-capacity", type=_positive_int, default=2)
    parser.add_argument("--perception-only", action="store_true")
    parser.add_argument("--duration", type=_positive_float, default=5.0)
    return parser.parse_args(argv)


def _start_pipelines(runtime: VehicleMindRuntime, args: argparse.Namespace):
    cabin = build_cabin_pipeline(
        runtime,
        video_path=args.cabin_video,
        model_path=args.cabin_model,
        inference_hz=args.cabin_hz,
        frame_capacity=args.frame_capacity,
    )
    cabin.start()
    try:
        road = build_road_pipeline(
            runtime,
            video_path=args.road_video,
            model_path=args.road_model,
            inference_hz=args.road_hz,
            frame_capacity=args.frame_capacity,
        )
        road.start()
    except Exception:
        cabin.stop()
        cabin.join(timeout=5)
        raise
    return cabin, road


def _pipeline_errors(pipelines: tuple) -> list[str]:
    return [
        f"{health.get('name', 'pipeline')}: {health['last_error']}"
        for pipeline in pipelines
        if (health := pipeline.health()).get("last_error")
    ]


def main(argv: list[str] | None = None) -> int:

    args = parse_args(argv)

    load_dotenv()

    print()
    print("========================================")

    print(" VehicleMind 舱内外与车机协同演示")

    print(" 舱内感知 + 道路感知 + 统一上下文 + Agent")

    print("========================================")

    llm = (
        ScriptedLLMClient((ScriptedResponse(content="unused"),))
        if args.perception_only
        else build_llm_client()
    )

    runtime = VehicleMindRuntime(llm=llm)

    # The vehicle state is simulated until a vehicle interface is available.
    runtime.context_manager.update_vehicle(
        speed_kmh=68.0,
        gear=GearState.D,
        cabin_temperature_c=28.0,
        target_temperature_c=24.0,
        ac_enabled=True,
        volume=25,
    )

    cabin_pipeline, road_pipeline = _start_pipelines(runtime, args)

    print()
    print("[VehicleMind] 等待感知结果...")

    time.sleep(2.0)

    print()
    print("[VehicleMind] 已就绪。")

    print("可用命令：")

    print("  context  - 查看统一上下文")

    print("  fresh    - 查看感知新鲜度")

    print("  events   - 查看近期语义事件")

    print("  health   - 查看流水线健康状态与延迟")

    print("  quit     - 退出")

    # Perception runs independently; the LLM is called only after user input.
    try:
        if args.perception_only:
            time.sleep(args.duration)
            for pipeline in (cabin_pipeline, road_pipeline):
                print(format_health(pipeline.health()))
            errors = _pipeline_errors((cabin_pipeline, road_pipeline))
            if errors:
                print("[VehicleMind 错误]", "; ".join(errors))
                return 1
            return 0
        while True:
            errors = _pipeline_errors((cabin_pipeline, road_pipeline))
            if errors:
                print("[VehicleMind 错误]", "; ".join(errors))
                return 1
            print()

            try:
                text = input("你 > ").strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):
                print()
                break

            if not text:
                continue

            command = text.lower()

            if command in {
                "q",
                "quit",
                "exit",
            }:
                break

            if command == "context":
                print(format_context(runtime.context_manager.get_agent_context()))

                continue

            if command == "fresh":
                freshness = runtime.context_manager.freshness()

                print()
                print(format_freshness(freshness))

                continue

            if command == "events":
                events = runtime.event_bus.recent_events(limit=20)

                if not events:
                    print("暂无事件。")

                else:
                    for event in events:
                        print(format_event(event))

                continue

            if command == "health":
                for pipeline in (cabin_pipeline, road_pipeline):
                    print(format_health(pipeline.health()))
                continue

            try:
                answer = runtime.chat(
                    text,
                    debug=True,
                )

            except Exception as exc:
                print()
                print("[VehicleMind 错误]")

                print(
                    type(exc).__name__,
                    str(exc),
                )

                continue

            print()
            print(
                "VehicleMind >",
                answer,
            )

    finally:
        print()
        print("[VehicleMind] 正在停止感知流水线...")

        cabin_pipeline.stop()
        road_pipeline.stop()
        cabin_pipeline.join(timeout=5.0)
        road_pipeline.join(timeout=5.0)

        print("[VehicleMind] 已完成退出。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

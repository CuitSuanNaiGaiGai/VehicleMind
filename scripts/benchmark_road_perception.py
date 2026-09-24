"""Measure YOLOPv2 road perception from a local video without video encoding."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from modules.driving.benchmark.runner import BenchmarkConfig, run_benchmark


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须为正整数") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="道路感知离线性能基准")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "coreml"), required=True)
    parser.add_argument("--work-width", type=_positive_int, default=1280)
    parser.add_argument("--work-height", type=_positive_int, default=720)
    parser.add_argument("--warmup-frames", type=_positive_int, default=5)
    parser.add_argument("--measure-frames", type=_positive_int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = BenchmarkConfig(
        video=args.video,
        model=args.model,
        provider=args.provider,
        work_width=args.work_width,
        work_height=args.work_height,
        warmup_frames=args.warmup_frames,
        measure_frames=args.measure_frames,
        output_dir=args.output_dir,
    )
    try:
        result = run_benchmark(config)
    except (FileExistsError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(
            f"错误：基准运行失败（{type(error).__name__}）；请检查本地视频、模型及执行环境。",
            file=sys.stderr,
        )
        return 2
    print(f"完成：{config.output_dir / 'report.md'}")
    print(
        f"测量帧数：{len(result['samples'])}；吞吐：{result['throughput_fps']:.2f} FPS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

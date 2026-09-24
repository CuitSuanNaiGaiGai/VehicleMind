"""Run the unlabeled cabin/road video audit from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.perception_audit.runner import run_audit


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须为正整数") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="舱内/舱外无标注视频自动核验")
    parser.add_argument("--cabin-dir", type=Path, required=True, help="舱内视频目录")
    parser.add_argument("--road-dir", type=Path, required=True, help="舱外视频目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/perception_audit"),
        help="本地运行根目录",
    )
    parser.add_argument(
        "--sample-interval",
        type=_positive_int,
        default=30,
        help="结构化展示点间隔；推理仍逐帧执行",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = run_audit(
            args.cabin_dir,
            args.road_dir,
            args.output_root,
            sample_interval=args.sample_interval,
        )
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as exc:
        print(
            f"错误：模型或感知依赖不可用（{type(exc).__name__}）。"
            "请先安装 perception extra 并核对本地模型。",
            file=sys.stderr,
        )
        return 2
    except (ValueError, FileExistsError) as exc:
        print(
            f"错误：输入或运行配置无效（{type(exc).__name__}）。"
            "请核对视频目录、数量及输出目录。",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            f"错误：核验未完成（{type(exc).__name__}）。"
            "请检查未标记 .complete 的本地运行目录。",
            file=sys.stderr,
        )
        return 2
    print(f"核验完成：{output / 'report.html'}")
    print("精度状态：not_evaluated（无对齐真值标签）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

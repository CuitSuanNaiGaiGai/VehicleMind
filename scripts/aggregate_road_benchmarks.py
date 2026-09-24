"""Summarize three compatible, completed road-perception benchmark runs."""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any
from collections.abc import Sequence


def _identity(result: dict[str, Any]) -> tuple[object, ...]:
    config = result["config"]
    provenance = result["provenance"]
    environment = result["environment"]
    return (
        provenance["video"]["sha256"],
        provenance["model"]["sha256"],
        provenance["git_commit"],
        provenance["dirty"],
        *(
            environment[key]
            for key in (
                "system",
                "release",
                "machine",
                "python",
                "onnxruntime",
                "opencv",
                "numpy",
            )
        ),
        *(
            config[key]
            for key in (
                "provider",
                "work_width",
                "work_height",
                "warmup_frames",
                "measure_frames",
                "score_threshold",
                "nms_threshold",
            )
        ),
        tuple(result["active_providers"]),
    )


def aggregate_runs(paths: Sequence[Path]) -> str:
    if len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError("必须提供三个不同的运行结果")
    runs = []
    for path in paths:
        if path.name != "result.json" or not (path.parent / ".complete").is_file():
            raise ValueError("运行结果未完成")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    try:
        run_ids = [run["run_id"] for run in runs]
        process_ids = [run["process_id"] for run in runs]
    except KeyError as error:
        raise ValueError("缺少独立进程标识") from error
    if len(set(run_ids)) != 3 or len(set(process_ids)) != 3:
        raise ValueError("三份结果必须来自三个独立进程")
    if any(_identity(run) != _identity(runs[0]) for run in runs[1:]):
        raise ValueError("三次运行的输入、模型或配置不一致")

    config = runs[0]["config"]
    lines = [
        "# 道路感知三次运行汇总",
        "",
        f"模式：`{config['provider']}`；工作尺寸：{config['work_width']}×{config['work_height']}。",
        "以下为三次独立进程测量；每次读取同一视频帧，不是三组独立场景。",
        "",
        "| 运行 | 推理 p50 / p95 (ms) | 完整帧 p50 / p95 (ms) | 吞吐 FPS | 峰值 RSS (MiB) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for number, run in enumerate(runs, 1):
        inference = run["summary"]["inference_ms"]
        frame = run["summary"]["frame_total_ms"]
        lines.append(
            f"| {number} | {inference['p50']:.2f} / {inference['p95']:.2f} | "
            f"{frame['p50']:.2f} / {frame['p95']:.2f} | "
            f"{run['throughput_fps']:.2f} | {run['environment']['peak_rss_mib']:.1f} |"
        )
    lines.extend(["", "跨运行范围（最小–最大）："])
    for label, values in (
        ("推理 p50 (ms)", [run["summary"]["inference_ms"]["p50"] for run in runs]),
        ("推理 p95 (ms)", [run["summary"]["inference_ms"]["p95"] for run in runs]),
        ("完整帧 p50 (ms)", [run["summary"]["frame_total_ms"]["p50"] for run in runs]),
        ("完整帧 p95 (ms)", [run["summary"]["frame_total_ms"]["p95"] for run in runs]),
        ("吞吐 FPS", [run["throughput_fps"] for run in runs]),
        ("峰值 RSS (MiB)", [run["environment"]["peak_rss_mib"] for run in runs]),
    ):
        lines.append(f"- {label}：{min(values):.2f}–{max(values):.2f}")
    lines.extend(
        [
            "",
            "本结果非感知精度；完整帧不含绘制、显示或视频编码，不能证明上车实时性能。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="聚合三次道路感知基准结果")
    parser.add_argument("results", nargs=3, type=Path)
    args = parser.parse_args(argv)
    try:
        print(aggregate_runs(args.results), end="")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(
            2, f"聚合失败：{type(error).__name__}；请检查三份完整结果及配置。\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

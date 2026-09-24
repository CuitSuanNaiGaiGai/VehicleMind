"""Aggregate unlabeled audit outputs and render privacy-safe Chinese reports."""

from __future__ import annotations

from collections import Counter
from html import escape

SAFE_FAILURES = frozenset({
    "视频无法打开", "视频元数据无有效帧", "首帧解码失败",
    "视频无有效帧", "视频提前结束，实际帧数少于媒体元数据",
})


def _failure_reason(value: str | None) -> str:
    if value in SAFE_FAILURES:
        return value
    if value and value.endswith(": 视频处理失败"):
        return "视频处理失败"
    return "其他失败"


def build_summary(catalog: dict, results: list[dict]) -> dict:
    """Use attempted videos (including failures) and processed frames separately."""
    domains = {}
    for domain in ("cabin", "road"):
        selected = [row for row in results if row["domain"] == domain]
        successes = sum(row["status"] == "success" for row in selected)
        processed = sum(int(row["processed_frames"]) for row in selected)
        valid = sum(int(row["valid_output_frames"]) for row in selected)
        failures = Counter(_failure_reason(row.get("error")) for row in selected
                           if row["status"] != "success")
        row = {
            "videos": {"successful": successes,
                       "attempted": catalog["attempted"][domain]},
            "frames": {"valid_output": valid, "processed": processed},
            "failures": dict(sorted(failures.items())),
        }
        if domain == "cabin":
            states = Counter()
            for video in selected:
                states.update(video.get("state_counts", {}))
            row["output"] = {
                "state_counts": dict(sorted(states.items())),
                "face_visible_frames": sum(video.get("face_visible_frames", 0)
                                           for video in selected),
                "eye_closed_frames": sum(video.get("eye_closed_frames", 0)
                                         for video in selected),
                "yawn_output_frames": sum(video.get("yawn_output_frames", 0)
                                          for video in selected),
            }
        else:
            objects = Counter()
            for video in selected:
                objects.update(video.get("object_counts", {}))
            row["output"] = {
                "object_counts": dict(sorted(objects.items())),
                "lane_detected_frames": sum(video.get("lane_detected_frames", 0)
                                            for video in selected),
                "drivable_detected_frames": sum(
                    video.get("drivable_detected_frames", 0) for video in selected),
                "lane_output_flips": sum(video.get("lane_output_flips", 0)
                                         for video in selected),
                "drivable_output_flips": sum(
                    video.get("drivable_output_flips", 0) for video in selected),
            }
        domains[domain] = row
    return {
        "domains": domains,
        "accuracy": {"status": "not_evaluated", "reason": "无对齐真值标签"},
        "limitations": [
            "仅统计模型输出与处理覆盖，不代表感知正确率。",
            "视频可能来自相同场景或驾驶员，帧数不等于独立样本数。",
            "来源与许可未核定，不公开原视频或逐视频记录。",
        ],
    }


def render_markdown(summary: dict) -> str:
    lines = ["# 感知自动核验报告", "", "无人工或官方对齐标签；精度状态："
             f"`{summary['accuracy']['status']}`（{summary['accuracy']['reason']}）。", ""]
    for domain, label in (("cabin", "舱内"), ("road", "舱外")):
        row = summary["domains"][domain]
        videos, frames, output = row["videos"], row["frames"], row["output"]
        lines.extend([
            f"## {label}", "",
            f"- 处理成功：{videos['successful']} / {videos['attempted']} 条视频。",
            f"- 有效输出：{frames['valid_output']} / {frames['processed']} 帧。",
        ])
        if domain == "cabin":
            lines.append(f"- 状态输出帧数：{output['state_counts']}。")
            lines.append(f"- 可见人脸 / 闭眼 / 哈欠输出："
                         f"{output['face_visible_frames']} / "
                         f"{output['eye_closed_frames']} / "
                         f"{output['yawn_output_frames']} 帧。")
        else:
            lines.append(f"- 目标输出次数（按帧累计）：{output['object_counts']}。")
            lines.append(f"- 车道 / 可行驶区输出："
                         f"{output['lane_detected_frames']} / "
                         f"{output['drivable_detected_frames']} 帧。")
            lines.append(f"- 相邻帧输出变化：车道 {output['lane_output_flips']} 次，"
                         f"可行驶区 {output['drivable_output_flips']} 次。")
        if row["failures"]:
            lines.append(f"- 失败类型与条数：{row['failures']}。")
        lines.append("")
    lines.extend(["## 解释边界", ""])
    lines.extend(f"- {limit}" for limit in summary["limitations"])
    return "\n".join(lines) + "\n"


def render_html(summary: dict) -> str:
    body = escape(render_markdown(summary))
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<title>感知自动核验报告</title><style>body{font:16px/1.7 system-ui;"
        "max-width:900px;margin:40px auto;padding:0 20px;color:#203044}"
        "pre{white-space:pre-wrap;background:#f4f7fb;padding:24px;border-radius:12px}"
        "</style></head><body><pre>" + body + "</pre></body></html>"
    )

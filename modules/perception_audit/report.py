"""Aggregate unlabeled audit outputs and render privacy-safe Chinese reports."""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from urllib.parse import quote

SAFE_FAILURES = frozenset(
    {
        "视频无法打开",
        "视频元数据无有效帧",
        "首帧解码失败",
        "视频无有效帧",
        "视频提前结束，实际帧数少于媒体元数据",
    }
)


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
        failures = Counter(
            _failure_reason(row.get("error"))
            for row in selected
            if row["status"] != "success"
        )
        row = {
            "videos": {
                "successful": successes,
                "attempted": catalog["attempted"][domain],
            },
            "frames": {"valid_output": valid, "processed": processed},
            "failures": dict(sorted(failures.items())),
        }
        if domain == "cabin":
            states = Counter()
            durations = Counter()
            for video in selected:
                states.update(video.get("state_counts", {}))
                durations.update(video.get("state_duration_seconds", {}))
            row["output"] = {
                "state_counts": dict(sorted(states.items())),
                "state_duration_seconds": dict(sorted(durations.items())),
                "face_visible_frames": sum(
                    video.get("face_visible_frames", 0) for video in selected
                ),
                "eye_closed_frames": sum(
                    video.get("eye_closed_frames", 0) for video in selected
                ),
                "yawn_output_frames": sum(
                    video.get("yawn_output_frames", 0) for video in selected
                ),
            }
        else:
            objects = Counter()
            for video in selected:
                objects.update(video.get("object_counts", {}))
            row["output"] = {
                "object_counts": dict(sorted(objects.items())),
                "lane_detected_frames": sum(
                    video.get("lane_detected_frames", 0) for video in selected
                ),
                "drivable_detected_frames": sum(
                    video.get("drivable_detected_frames", 0) for video in selected
                ),
                "lane_output_flips": sum(
                    video.get("lane_output_flips", 0) for video in selected
                ),
                "drivable_output_flips": sum(
                    video.get("drivable_output_flips", 0) for video in selected
                ),
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
    lines = [
        "# 感知自动核验报告",
        "",
        "无人工或官方对齐标签；精度状态："
        f"`{summary['accuracy']['status']}`（{summary['accuracy']['reason']}）。",
        "",
    ]
    for domain, label in (("cabin", "舱内"), ("road", "舱外")):
        row = summary["domains"][domain]
        videos, frames, output = row["videos"], row["frames"], row["output"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- 处理成功：{videos['successful']} / {videos['attempted']} 条视频。",
                f"- 有效输出：{frames['valid_output']} / {frames['processed']} 帧。",
            ]
        )
        if domain == "cabin":
            lines.append(f"- 状态输出帧数：{output['state_counts']}。")
            lines.append(
                f"- 状态输出时长（秒，按视频时间戳近似）："
                f"{output['state_duration_seconds']}。"
            )
            lines.append(
                f"- 可见人脸 / 闭眼 / 哈欠输出："
                f"{output['face_visible_frames']} / "
                f"{output['eye_closed_frames']} / "
                f"{output['yawn_output_frames']} 帧。"
            )
        else:
            lines.append(f"- 目标输出次数（按帧累计）：{output['object_counts']}。")
            lines.append(
                f"- 车道 / 可行驶区输出："
                f"{output['lane_detected_frames']} / "
                f"{output['drivable_detected_frames']} 帧。"
            )
            lines.append(
                f"- 相邻帧输出变化：车道 {output['lane_output_flips']} 次，"
                f"可行驶区 {output['drivable_output_flips']} 次。"
            )
        if row["failures"]:
            lines.append(f"- 失败类型与条数：{row['failures']}。")
        lines.append("")
    lines.extend(["## 解释边界", ""])
    lines.extend(f"- {limit}" for limit in summary["limitations"])
    return "\n".join(lines) + "\n"


def render_html(
    summary: dict,
    *,
    results: list[dict] | None = None,
    video_links: dict[str, str] | None = None,
) -> str:
    def metric(label: str, numerator: int, denominator: int, unit: str) -> str:
        ratio = 100 * numerator / denominator if denominator else 0
        ratio_label = f"{ratio:.1f}% · 分子 / 分母" if denominator else "无可计算比例"
        return (
            f'<article class="metric"><span>{escape(label)}</span>'
            f"<strong>{numerator}<small> / {denominator} {unit}</small></strong>"
            f'<div class="track"><i style="width:{ratio:.1f}%"></i></div>'
            f"<em>{ratio_label}</em></article>"
        )

    cards = []
    rows = []
    for domain, label in (("cabin", "舱内"), ("road", "舱外")):
        data = summary["domains"][domain]
        videos, frames, output = data["videos"], data["frames"], data["output"]
        cards.append(
            f'<section class="domain"><div class="section-title"><span class="chip">'
            f"{escape(label)}</span><h2>{escape(label)}感知</h2></div>"
            '<section class="metric-grid">'
            + metric(
                "视频处理成功", videos["successful"], videos["attempted"], "条视频"
            )
            + metric(
                "有效结构化输出", frames["valid_output"], frames["processed"], "帧"
            )
            + "</section></section>"
        )
        if domain == "cabin":
            details = (
                f"驾驶员状态分布：{output['state_counts']}；"
                f"状态时长（秒，近似）：{output['state_duration_seconds']}；"
                f"人脸可见 {output['face_visible_frames']} 帧、"
                f"闭眼 {output['eye_closed_frames']} 帧、"
                f"哈欠输出 {output['yawn_output_frames']} 帧"
            )
        else:
            details = (
                f"目标输出（按帧累计）：{output['object_counts']}；"
                f"车道 {output['lane_detected_frames']} 帧、"
                f"可行驶区 {output['drivable_detected_frames']} 帧；"
                f"相邻帧输出变化：车道 {output['lane_output_flips']} 次、"
                f"可行驶区 {output['drivable_output_flips']} 次"
            )
        failures = (
            "、".join(f"{name} {count} 条" for name, count in data["failures"].items())
            or "无"
        )
        rows.append(
            f"<tr><th>{escape(label)}</th><td>{escape(details)}</td>"
            f"<td>{escape(failures)}</td></tr>"
        )
    limitations = "".join(f"<li>{escape(text)}</li>" for text in summary["limitations"])
    per_video = ""
    if results is not None:
        video_rows = []
        for row in results:
            if row["domain"] == "cabin":
                detail = (
                    f"驾驶员状态帧数：{row.get('state_counts', {})}；"
                    f"状态时长（秒，近似）：{row.get('state_duration_seconds', {})}"
                )
            else:
                detail = f"目标输出（按帧累计）：{row.get('object_counts', {})}"
            status = (
                "完成"
                if row["status"] == "success"
                else _failure_reason(row.get("error"))
            )
            links = []
            relative = (video_links or {}).get(row["id"])
            if relative and not Path(relative).is_absolute():
                for point in row.get("samples", [])[:3]:
                    seconds = point["timestamp_ms"] / 1000
                    href = quote(relative, safe="/.") + f"#t={seconds:g}"
                    links.append(
                        f'<a href="{escape(href, quote=True)}">{seconds:g} 秒</a>'
                    )
            if links:
                detail += "；本地视频时间点："
            video_rows.append(
                f"<tr><th>{escape(row['id'])}</th><td>{escape(status)}</td>"
                f"<td>{row['valid_output_frames']} / {row['processed_frames']} 帧</td>"
                f"<td>{escape(detail)}{'、'.join(links)}</td></tr>"
            )
        per_video = (
            '<section class="detail"><h2>逐视频核对</h2>'
            "<p>仅本地运行报告展示文件名、结构化输出和视频时间点；"
            "链接依赖本机原视频，不含原始画面。</p>"
            "<table><thead><tr><th>视频 ID</th><th>处理状态</th>"
            "<th>有效输出 / 处理帧</th><th>输出摘要</th></tr></thead><tbody>"
            + "".join(video_rows)
            + "</tbody></table></section>"
        )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>感知自动核验报告 · VehicleMind</title><style>"
        ":root{color-scheme:light}*{box-sizing:border-box}"
        "body{margin:0;background:#f3f7fb;color:#17304a;font:16px/1.65 "
        'system-ui,-apple-system,"PingFang SC",sans-serif}'
        ".layout{max-width:1050px;margin:auto;padding:50px 24px 72px}"
        ".eyebrow{letter-spacing:.16em;color:#247cac;font-size:12px;font-weight:700}"
        "h1{font-size:clamp(30px,5vw,48px);line-height:1.2;margin:10px 0}"
        "h2{margin:0;font-size:24px}.intro{color:#51677d;max-width:780px}"
        ".notice,.domain,.detail{background:#fff;border:1px solid #dce7ef;"
        "border-radius:18px;box-shadow:0 10px 30px #1f557010;padding:24px;margin-top:22px}"
        ".notice{border-left:5px solid #d69a24;background:#fffcf4}"
        ".notice strong{color:#895900}.section-title{display:flex;align-items:center;gap:12px}"
        ".chip{background:#e3f3fb;color:#126b94;border-radius:999px;padding:3px 11px;"
        "font-size:13px;font-weight:700}.metric-grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:18px}"
        ".metric{background:#f5f9fc;border-radius:14px;padding:18px;display:flex;"
        "flex-direction:column;gap:5px}.metric span{color:#526a7f;font-size:14px}"
        ".metric strong{font-size:32px;line-height:1.2}.metric small{font-size:14px;"
        "font-weight:500;color:#60778b}.metric em{font-style:normal;font-size:12px;"
        "color:#60778b}.track{height:7px;background:#dce9f1;border-radius:9px;"
        "overflow:hidden;margin:8px 0}.track i{display:block;height:100%;"
        "background:linear-gradient(90deg,#1d87bd,#30b6a5)}"
        "table{width:100%;border-collapse:collapse;margin-top:14px}th,td{padding:14px;"
        "text-align:left;border-bottom:1px solid #e5edf3;vertical-align:top}"
        "th{white-space:nowrap;color:#126b94}td{overflow-wrap:anywhere}"
        ".foot{color:#65798b;font-size:13px;margin-top:28px}"
        "@media(max-width:650px){.layout{padding:26px 14px}table{font-size:14px}}"
        '</style></head><body><main class="layout">'
        '<div class="eyebrow">VEHICLEMIND · 本地私有核验</div>'
        "<h1>感知自动核验报告</h1>"
        '<p class="intro">离线视频逐帧推理的处理覆盖与输出行为。舱内、舱外分别统计；'
        "视频与帧不混用为同一分母。</p>"
        '<aside class="notice"><strong>精度状态：not_evaluated</strong><br>'
        "当前视频缺少对齐真值标签；以下数字不能解释为检测准确率、误报率或漏报率。"
        "</aside>"
        + "".join(cards)
        + '<section class="detail"><h2>输出概览与失败类型</h2><table>'
        "<thead><tr><th>域</th><th>模型输出行为</th><th>处理失败</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
        + per_video
        + '<section class="detail"><h2>解释边界</h2><ul>'
        + limitations
        + '</ul></section><p class="foot">逐视频结构化记录与哈希仅保存在本地运行目录；'
        "不自动上传画面、视频或个人轨迹。</p></main></body></html>"
    )

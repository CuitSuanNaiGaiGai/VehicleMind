"""Aggregate cabin timings and render compact bilingual performance metrics."""

from __future__ import annotations

import math
from html import escape

from modules.perception_audit.performance import summarize_ms

LATENCY_LABELS = (
    ("service_init_ms", "Service Initialization Latency (服务初始化延迟)"),
    ("first_frame_ms", "First-frame Latency (首帧处理延迟)"),
    ("steady_frame_ms", "Steady-state Frame Latency (稳态单帧处理延迟)"),
)
THROUGHPUT_LABEL = "Offline Replay Throughput (离线回放吞吐)"
NO_DATA = "暂无可计算性能数据"


def build_performance(rows: list[dict]) -> dict:
    """Use successful videos, with matching frame/time coverage for throughput."""
    successful = [row for row in rows if row["status"] == "success"]
    timings = [row.get("performance", {}) for row in successful]
    replay_rows = []
    for row in successful:
        elapsed = row.get("performance", {}).get("replay_loop_ms")
        if elapsed is not None and math.isfinite(elapsed) and elapsed > 0:
            replay_rows.append((int(row["processed_frames"]), float(elapsed)))
    replay_frames = sum(frames for frames, _ in replay_rows)
    replay_ms = sum(elapsed for _, elapsed in replay_rows)
    return {
        "video_count": len(successful),
        "processed_frames": sum(int(row["processed_frames"]) for row in successful),
        "service_init_ms": summarize_ms(
            [
                row["service_init_ms"]
                for row in timings
                if row.get("service_init_ms") is not None
            ]
        ),
        "first_frame_ms": summarize_ms(
            [
                row["first_frame_ms"]
                for row in timings
                if row.get("first_frame_ms") is not None
            ]
        ),
        "steady_frame_ms": summarize_ms(
            [
                sample
                for row in timings
                for sample in row.get("steady_frame_ms_samples", [])
            ]
        ),
        "replay_video_count": len(replay_rows),
        "replay_processed_frames": replay_frames,
        "replay_fps": replay_frames / (replay_ms / 1000) if replay_rows else None,
    }


def _metrics(performance: dict | None) -> list[tuple[str, str]]:
    data = performance or {}
    metrics = []
    for key, label in LATENCY_LABELS:
        stats = data.get(key)
        value = (
            f"n={stats['count']} · p50={stats['p50']:.2f} ms · p95={stats['p95']:.2f} ms"
            if stats
            else NO_DATA
        )
        metrics.append((label, value))
    fps = data.get("replay_fps")
    throughput = (
        f"{fps:.2f} FPS · n={data['replay_video_count']} 条视频 · "
        f"{data['replay_processed_frames']} 帧"
        if fps is not None
        else NO_DATA
    )
    return [*metrics, (THROUGHPUT_LABEL, throughput)]


def render_markdown(performance: dict | None) -> list[str]:
    return [f"- {label}：{value}。" for label, value in _metrics(performance)]


def render_html(performance: dict | None) -> str:
    rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
        for label, value in _metrics(performance)
    )
    return (
        '<section class="detail"><h2>舱内性能</h2>'
        "<p>初始化与首帧按视频统计；稳态延迟按首帧后的处理帧统计。"
        "吞吐按有有效计时的成功视频统计，包含解码与核验记录开销。</p>"
        "<table><thead><tr><th>指标</th><th>聚合结果</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></section>"
    )

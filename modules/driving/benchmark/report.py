"""Chinese, source-grounded report for one road benchmark run."""

from __future__ import annotations

from collections.abc import Mapping


STAGES = (
    ("video_read_ms", "视频读取"),
    ("resize_ms", "尺寸调整"),
    ("preprocess_ms", "检测器预处理"),
    ("inference_ms", "ONNX 推理"),
    ("postprocess_ms", "检测器后处理"),
    ("detector_total_ms", "检测器整体"),
    ("frame_total_ms", "完整帧"),
)


def render_report(result: Mapping[str, object]) -> str:
    """Render only fields present in the sanitized result mapping."""

    config = result["config"]
    provenance = result["provenance"]
    timing = result["timing"]
    environment = result["environment"]
    summary = result["summary"]
    assert isinstance(config, Mapping)
    assert isinstance(provenance, Mapping)
    assert isinstance(timing, Mapping)
    assert isinstance(environment, Mapping)
    assert isinstance(summary, Mapping)
    video = provenance["video"]
    model = provenance["model"]
    providers = result["active_providers"]
    assert isinstance(video, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(providers, list) and all(
        isinstance(item, str) for item in providers
    )

    lines = [
        "# 道路感知离线性能测量",
        "",
        f"- 视频：`{video['name']}`（SHA-256 `{video['sha256']}`）",
        f"- 模型：`{model['name']}`（SHA-256 `{model['sha256']}`）",
        f"- 配置：{config['work_width']}×{config['work_height']}；预热 {config['warmup_frames']} 帧；测量 {config['measure_frames']} 帧",
        f"- 请求模式：`{config['provider']}`；实际 session providers：`{', '.join(providers)}`",
        f"- 环境：{environment['system']} {environment['release']} / {environment['machine']}；Python {environment['python']}；ONNX Runtime {environment['onnxruntime']}",
        f"- Git：`{provenance['git_commit']}`；dirty={provenance['dirty']}",
        f"- CoreML 缓存运行前存在：{provenance['coreml_cache_existed']}",
        "",
        "| 阶段 | n | 均值 ms | p50 ms | p95 ms | 最小 ms | 最大 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in STAGES:
        stage = summary[key]
        assert isinstance(stage, Mapping)
        lines.append(
            f"| {label} | {stage['count']} | {stage['mean']:.2f} | {stage['p50']:.2f} | {stage['p95']:.2f} | {stage['min']:.2f} | {stage['max']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"会话初始化：{timing['session_init_ms']:.2f} ms；首帧：{timing['cold_first_frame_ms']:.2f} ms。",
            f"测量阶段吞吐：{result['throughput_fps']:.2f} FPS；进程峰值 RSS：{environment['peak_rss_mib']:.1f} MiB。",
            "",
            "本结果是同一短视频上的本机测量，不含视频绘制或编码；完整帧包括读取、尺寸调整和检测。",
            "首帧属于预热，不进入 p50/p95。会话初始化在当前 CoreML 缓存状态下测得，不等于首次编译耗时。",
            "CoreML 优先配置可能有算子回退 CPU；进程峰值 RSS 不是模型独占内存。本报告非感知精度或通用实时性能保证。",
            "",
        ]
    )
    return "\n".join(lines)

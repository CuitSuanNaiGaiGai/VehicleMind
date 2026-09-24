# Road Perception Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用本地离线道路视频与 YOLOPv2 权重生成可追溯、可核验、没有虚构 provider 或精度数字的性能报告。

**Architecture:** 统计逻辑与模型/视频 IO 分离。基准运行器调用现有 `PanopticDrivingDetector.detect()` 的分段计时，另计解码、尺寸调整和完整帧；CLI 生成原子化 JSON 与中文 Markdown，聚合器只读取 JSON。模型和视频不进 Git。

**Tech Stack:** Python 3.13、OpenCV、NumPy、ONNX Runtime、pytest、Ruff；本机运行可选 CoreMLExecutionProvider。

## Global Constraints

- 完整规格见 `docs/superpowers/specs/2026-09-24-road-perception-benchmark-design.md`；本计划只实现道路路径，舱内随后单独实施。
- 默认工作尺寸 1280×720，预热 5 帧、测量 30 帧；正式报告同一输入与配置下每组独立进程运行 3 次。
- CoreML 不可用或实际 session 未列出 `CoreMLExecutionProvider` 时不得输出 CoreML 成绩；即使列出，也称“CoreML 优先配置”，不声称纯 CoreML。
- `frame_total_ms` 包含视频读取、尺寸调整和检测，不含绘制、显示或编码；FPS 用测量阶段墙钟时间计算。
- 不提交原始视频、模型、运行目录或本机绝对路径。结果只称本机短视频性能，不称精度、实时上车或通用性能。
- 在当前 `codex/road-perception-benchmark-design` 分支执行；每个可独立验证的任务完成后提交、推送并核对 `todolist.md`。

---

### Task 1: 纯统计与结果契约

**Files:** Create `modules/driving/benchmark/__init__.py`, `modules/driving/benchmark/stats.py`, `tests/driving/benchmark/test_stats.py`.

**Interfaces:** `summarize_ms(values: Sequence[float]) -> dict[str, float | int]` 返回 `count/mean/p50/p95/min/max`；`throughput_fps(frames: int, elapsed_seconds: float) -> float` 使用墙钟。输入为空、非有限或负值均抛 `ValueError`。

- [x] **Step 1: 写失败测试。** 在 `tests/driving/benchmark/test_stats.py` 写：

```python
import pytest
from modules.driving.benchmark.stats import summarize_ms, throughput_fps


def test_linear_percentiles_and_wall_clock_fps() -> None:
    summary = summarize_ms([10.0, 20.0, 30.0, 40.0])
    assert summary == {"count": 4, "mean": 25.0, "p50": 25.0, "p95": 38.5, "min": 10.0, "max": 40.0}
    assert throughput_fps(30, 2.0) == 15.0


@pytest.mark.parametrize("values", [[], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_samples_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        summarize_ms(values)
```

- [x] **Step 2: 验证红灯。** `uv run --group dev python -m pytest -q tests/driving/benchmark/test_stats.py`；预期模块不存在导致失败。
- [x] **Step 3: 实现。** 在 `stats.py` 用 `sorted(values)`、`statistics.fmean` 和位置 `(n-1)*q` 的线性插值求 `p50/p95`；`throughput_fps` 拒绝非正帧数或时间，不取各帧 FPS 的平均值。保留上述公开函数签名。
- [x] **Step 4: 验证绿灯并提交。** 目标测试、Ruff、`git diff --check` 均通过；提交 `Add road benchmark statistics` 并推送。

### Task 2: 无视频编码的道路测量与原子产物

**Files:** Create `modules/driving/benchmark/runner.py`, `modules/driving/benchmark/report.py`, `scripts/benchmark_road_perception.py`, `tests/driving/benchmark/test_runner.py`.

**Interfaces:** `BenchmarkConfig(video: Path, model: Path, provider: Literal["cpu", "coreml"], work_width: int = 1280, work_height: int = 720, warmup_frames: int = 5, measure_frames: int = 30, output_dir: Path)`；`run_benchmark(config, detector_factory=None, capture_factory=None) -> dict[str, object]`。默认 factory 延迟导入 `PanopticDrivingDetector`，以 `warmup_runs=0` 初始化；测试注入伪 detector/capture，不要求 ONNX Runtime。

- [x] **Step 1: 写失败测试。** 在 `tests/driving/benchmark/test_runner.py` 用 7 帧的伪 capture、带固定 `DrivingSceneResult` 时间字段的伪 detector：预热 2 帧、测量 4 帧时结果必须只含 4 条逐帧时间；第 7 帧不消费。另测 5 帧输入却要求预热 2 + 测量 4 时抛 `ValueError` 且输出目录不存在。用伪 session `get_providers()` 分别验证 CoreML 缺失时拒绝和 CPU 组无误报。
- [x] **Step 2: 验证红灯。** `uv run --group dev python -m pytest -q tests/driving/benchmark/test_runner.py`；预期 `runner`/`report` 尚不存在。
- [x] **Step 3: 实现运行器。** 验证输入文件、正整数参数和输出目录未存在；在同一进程按视频顺序读取第一帧作 `cold_first_frame_ms`，再读完预热帧、测量帧；实现时把首帧计入预热 5 帧总数，绝不混入测量统计。调用 `detector.detect(resized_frame)`，把每帧 `video_read_ms/resize_ms/preprocess_ms/inference_ms/postprocess_ms/detector_total_ms/frame_total_ms` 写入结果。测量墙钟从第一条测量帧读取前到最后一条处理后。始终 release capture。
- [x] **Step 4: 实现来源与运行环境。** `result.json` 只写输入 basename、大小、SHA-256，不写绝对路径；包含 Git commit/dirty、实际 session provider 列表、CoreML 缓存是否已存在、session 初始化时长、冷首帧时长、Python/OS/芯片/OpenCV/NumPy/ONNX Runtime 版本、配置、峰值 RSS。macOS 的 `ru_maxrss` 按字节、Linux 按 KiB 转成 MiB；报告注明进程级峰值。CoreML 组未激活指定 provider 时直接失败且不落盘。
- [x] **Step 5: 原子写入。** 在目标目录同级建临时目录，先生成 `result.json` 和由同一 dict 渲染的中文 `report.md`，成功后 rename 到目标；异常时只移除本次创建的临时目录。`scripts/benchmark_road_perception.py` 解析规格列出的参数，捕获已知输入/运行错误并给中文错误，不输出本机完整路径。
- [x] **Step 6: 验证与提交。** 目标测试至少覆盖短视频、provider、统计样本、字段脱敏、目标目录已存在、原子失败；运行全量离线 pytest、Ruff、源码大小和 `git diff --check`。提交 `Build reproducible road benchmark runner` 并推送。

### Task 3: 三次实测、聚合与受限展示

**Files:** Create `scripts/aggregate_road_benchmarks.py`, `tests/driving/benchmark/test_aggregate.py`, `docs/reports/2026-09-24-road-performance.md`; modify `README.md`, `todolist.md`.

**Interfaces:** 聚合命令接收 3 个同 provider 的 `result.json`；校验视频哈希、模型哈希、工作尺寸、预热/测量帧数、阈值和实际 provider 列表一致；输出中文三次区间及结果表，不把同视频 90 帧称作 90 条独立样本。

- [ ] **Step 1: 写失败测试。** 三个同配置结果可聚合；模型哈希或 provider 不同必须拒绝；报告明确包含“三次独立进程”“同一视频帧”“非感知精度”。运行目标测试确认红灯。
- [ ] **Step 2: 实现聚合脚本并验证。** 从三个 JSON 读取每次 p50/p95、FPS 与 RSS；拒绝非 3 份或配置不一致输入；输出按运行编号的表及跨运行最小/最大值。跑目标测试、Ruff 和 `git diff --check`，提交 `Aggregate independent road benchmark runs` 并推送。
- [ ] **Step 3: 验证本机资产。** 运行 `uv run --group dev python scripts/verify_assets.py`；确认本地 `models/driving/YOLOPv2_512.onnx` 哈希匹配 manifest。若失败，停止真实测量，不替换资产或填数。
- [ ] **Step 4: 实测。** 对 `assets/driving/road_test.mp4` 和固定模型按 CPU、CoreML 优先配置各运行 3 个独立进程；每次使用新 `--output-dir`，输出留在 Git 忽略的 `runs/`。若 CoreML 不可用，只记录不可用并保留 CPU 三次；不静默回退。聚合前核查三个 JSON 的同源性与路径脱敏。
- [ ] **Step 5: 写公开报告。** 从 JSON 生成 `docs/reports/2026-09-24-road-performance.md`，列出设备、系统、版本、模型/视频哈希、分辨率、缓存状态、每组逐次和聚合区间、阶段 p50/p95、真实 provider、FPS 边界及样本局限。README 仅摘录有证据支撑的数字并链接报告；只勾选 `todolist.md` 第 6.3 节实际完成的道路性能项，舱内和感知精度保持待办。
- [ ] **Step 6: 总体验收。** 运行全部离线 pytest、Ruff check/format、mypy 当前 CI 边界、源码大小、`git diff --check`；审查公开文件无密钥、绝对路径和夸大表述。提交推送，创建 PR 并等待必需 CI；CI 绿灯后再讨论合并 main。

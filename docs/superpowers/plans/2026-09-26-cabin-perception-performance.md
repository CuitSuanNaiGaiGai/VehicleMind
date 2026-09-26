# 舱内感知处理性能基准实施计划

> **For agentic workers:** 使用 `executing-plans`（本项目已采用当前会话内联执行）按任务推进。步骤以复选框追踪。

**Goal:** 在现有无标签视频自动核验中增加有明确边界的舱内初始化、帧延迟和离线吞吐指标，并用本地固定视频生成可追溯报告。

**Architecture:** `process_video()` 记录每条舱内视频的服务初始化、首帧、稳态帧样本和回放循环墙钟耗时；独立统计模块负责有限值校验与分位数；现有 `build_summary()` 与中文 Markdown/HTML 展示聚合值。道路感知指标不改动，旧运行结果仍可读。

**Tech Stack:** Python 3.13、`time.perf_counter`、MediaPipe CabinPerceptionService metadata、pytest、现有视频自动核验与本地 `runs/` 结果结构。

## Global Constraints

- 不增加新的运行时依赖；不改动感知模型、阈值、Agent 决策或现有精度状态。
- `processing_ms` 是 `process_frame()` 单帧调用时间；回放循环吞吐包含视频读取与结果更新，不包含服务初始化、报告渲染和页面展示。
- 每条成功舱内视频的首帧独立报告；P50/P95 的稳态帧统计排除每条视频首帧。
- 汇总只使用完整成功视频；不允许空样本伪装成 0 延迟。历史结果缺少性能字段时仍能读取。
- 原始逐帧时延只写入 Git 忽略的本地 `runs/` 结果；公开文档只保存聚合值、运行/配置/模型版本和边界说明。
- 所有面向用户的报告为中文，指标采用 `English (中文含义)`；不标注真值准确率、不做人工标注、不做消融。
- 新增单个 Python 源码文件保持小于 300 行，不拆分无关模块。

---

### Task 1：时延统计契约

**Files:**
- Create: `modules/perception_audit/performance.py`
- Create: `tests/perception_audit/test_performance.py`

**Interface:** `summarize_ms(values: Sequence[float]) -> dict[str, float | int] | None`；空序列返回 `None`；负数、NaN、Infinity 抛 `ValueError`；P50/P95 使用 `(n-1) * q` 线性插值，与道路性能报告算法一致。

- [x] **Step 1: 先写失败测试**，覆盖空序列、`[10, 20, 40]` 得到 count 3 / mean 70÷3 / p50 20 / p95 38 / min 10 / max 40，以及负数和非有限值拒绝。
- [x] **Step 2: 运行定向测试确认 RED**：`uv run --group dev pytest tests/perception_audit/test_performance.py -q`；预期因模块不存在而导入失败。
- [x] **Step 3: 实现最小统计函数**：用 `math.isfinite` 校验，排序后以位置 `(n - 1) * q` 线性插值，空输入返回 `None`。

```python
def summarize_ms(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("timing samples must be finite and non-negative")

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }
```

- [x] **Step 4: 重跑定向测试确认 GREEN**，再运行 Ruff 与 `git diff --check`。

### Task 2：逐视频舱内计时

**Files:**
- Modify: `modules/perception_audit/process.py`
- Modify: `tests/perception_audit/test_process.py`

**Interface:** 舱内成功视频结果增加 `performance`：`service_init_ms`、`first_frame_ms`、`steady_frame_ms_samples`、`replay_loop_ms`。稳态样本从第二个处理帧开始；计时从 `snapshot.metadata.processing_ms` 读取，不用外层推测模型时间。道路结果结构保持原样。

- [x] **Step 1: 更新伪服务并写 RED 测试**：伪 cabin metadata 按帧返回 1/2/3ms；断言首帧为 1ms、稳态样本为 `[2, 3]`、初始化和回放墙钟非负；road 结果没有 cabin 性能字段；损坏视频仍标失败。
- [x] **Step 2: 运行定向测试确认 RED**：`uv run --group dev pytest tests/perception_audit/test_process.py -q`；失败应指向缺少性能字段。
- [x] **Step 3: 实现最小计时**：服务工厂调用外包 `perf_counter()` 计算初始化；只对 cabin 收集 metadata 时延；首帧与后续帧分开；从打开视频前到解码/处理循环结束计墙钟，排除模型初始化及 `service.close()`。

```python
if item["domain"] == "cabin":
    result["performance"] = {
        "service_init_ms": None,
        "first_frame_ms": None,
        "steady_frame_ms_samples": [],
        "replay_loop_ms": None,
    }

init_started = time.perf_counter()
service = service_factory(item["domain"])
if item["domain"] == "cabin":
    result["performance"]["service_init_ms"] = (
        time.perf_counter() - init_started
    ) * 1000

processing_ms = float(snapshot.metadata.processing_ms)
if item["domain"] == "cabin":
    if index == 0:
        result["performance"]["first_frame_ms"] = processing_ms
    else:
        result["performance"]["steady_frame_ms_samples"].append(processing_ms)

replay_loop_started = time.perf_counter()  # immediately before capture creation
result["performance"]["replay_loop_ms"] = (
    time.perf_counter() - replay_loop_started
) * 1000  # after capture.release(), immediately before service.close()
```

- [x] **Step 4: 重跑 process 测试**，确认时间样本对齐 `processed_frames`，失败结果不会被摘要误算为完整视频。

### Task 3：聚合与中文展示

**Files:**
- Modify: `modules/perception_audit/report.py`
- Modify: `tests/perception_audit/test_report.py`

- [x] **Step 1: 写 RED 测试**：成功与失败视频混合时，只汇总成功 cabin 行；检查服务初始化、首帧、稳态帧的样本数及分位数；旧结果缺失 `performance` 时 `build_summary()` 和 Markdown/HTML 不报错、显示“暂无可计算性能数据”。
- [x] **Step 2: 实现摘要**：通过 Task 1 的 `summarize_ms` 生成初始化、首帧、稳态帧的聚合；离线吞吐只按有有限正回放耗时的成功视频统计，并让对应处理帧数与耗时使用同一视频子集，避免旧结果缺失耗时时虚高；回放时间无有效样本时使用 `null`。

```python
successful = [row for row in cabin_rows if row["status"] == "success"]
performance = [row.get("performance", {}) for row in successful]
init_samples = [
    float(row["service_init_ms"])
    for row in performance
    if row.get("service_init_ms") is not None
]
first_samples = [
    float(row["first_frame_ms"])
    for row in performance
    if row.get("first_frame_ms") is not None
]
steady_samples = [
    float(sample)
    for row in performance
    for sample in row.get("steady_frame_ms_samples", [])
]
processed_frames = sum(row["processed_frames"] for row in successful)
replay_rows = [
    (int(video["processed_frames"]), float(perf["replay_loop_ms"]))
    for video in successful
    for perf in [video.get("performance", {})]
    if perf.get("replay_loop_ms") is not None
    and math.isfinite(float(perf["replay_loop_ms"]))
    and float(perf["replay_loop_ms"]) > 0
]
replay_frames = sum(frames for frames, _ in replay_rows)
replay_ms = sum(elapsed for _, elapsed in replay_rows)
summary = {
    "video_count": len(successful),
    "processed_frames": processed_frames,
    "service_init_ms": summarize_ms(init_samples),
    "first_frame_ms": summarize_ms(first_samples),
    "steady_frame_ms": summarize_ms(steady_samples),
    "replay_video_count": len(replay_rows),
    "replay_processed_frames": replay_frames,
    "replay_fps": replay_frames / (replay_ms / 1000)
    if replay_rows
    else None,
}
```

- [x] **Step 3: 实现中文报告**：Markdown 和 HTML 使用 `Service Initialization Latency (服务初始化延迟)`、`First-frame Latency (首帧处理延迟)`、`Steady-state Frame Latency (稳态单帧处理延迟)`、`Offline Replay Throughput (离线回放吞吐)`；呈现 n、p50/p95，HTML 不输出原始逐帧样本。
- [x] **Step 4: 重跑 report 与 perception audit 定向测试**，检查历史 schema 兼容、无绝对路径、无 `Precision/Recall/F1` 等准确率暗示。

### Task 4：真实固定样本与公开证据

**Files:**
- Create: `docs/reports/2026-09-26-cabin-performance.md`
- Modify: `README.md`
- Modify: `todolist.md`
- Modify: `modules/perception_audit/runner.py` and its provenance tests, to record the effective OpenCV runtime alongside installed distribution versions.

- [x] 用 perception extra 在本地现有舱内 26 条、舱外 26 条视频完整重跑审计；不上传视频/画面，不覆盖旧 run。检查新 manifest 记录 Git commit、配置哈希、依赖分发包版本（含 OpenCV 实际运行时）和模型哈希，且 `.complete` 存在。
- [x] 从新 `summary.json` 与 `manifest.json` 生成中文公开报告；只摘录真实聚合指标、n、硬件/软件/模型信息、计时口径和样本局限，不写本机路径或逐视频记录。
- [x] 更新 README 指向报告；仅勾选舱内性能及其可追溯性 TODO，不勾选准确率、类别语义或标签相关条目。
- [x] 运行完整离线测试 `uv run --group dev pytest -m "not hardware and not online" -q`、Ruff check/format、CI 范围 mypy、源码大小、资产清单 schema 与 `git diff --check`。
- [ ] 代码审查、提交并推送功能分支；通过 `core-quality` 后按仓库保护规则合并 main。

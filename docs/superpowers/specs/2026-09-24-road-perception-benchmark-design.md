# 道路感知离线性能基准设计

## 目标与边界

用本地离线道路视频和本地 YOLOPv2 ONNX 权重，生成能追溯输入、模型、环境与配置的**本机性能测量**。第一阶段只测道路推理；舱内管线随后按相同统计口径单独实现。性能报告不等于感知精度，不代表实时上车能力；不上传原始视频、模型权重或包含本机绝对路径的运行产物。

## 方案选择

现有 `apps/driving_demo/scene_runner.py` 的平均 FPS 包含绘制与视频编码，不能直接用于模型延迟声明。采用独立的只读基准入口，调用已有 `PanopticDrivingDetector.detect`，复用其 `preprocess_ms`、`inference_ms`、`postprocess_ms` 和 `total_ms`；外围另计视频读取与尺寸调整后的完整帧耗时。不开显示窗口、不写视频。外部通用 profiler 会增加环境差异，暂不作为主结果。

## 输入与运行契约

- 新 CLI 默认不绑定仓库内私有文件；显式接收 `--video`、`--model`、`--provider cpu|coreml`、`--work-width`、`--work-height`、`--warmup-frames`、`--measure-frames`、`--output-dir`。输入模型默认不下载。入口在模型/视频缺失或不可读时以中文错误结束，不生成半成品报告。
- 默认尺寸 1280×720、预热 5 帧、测量 30 帧；正式展示连续运行 3 个独立进程，每次从视频起点读取同样帧序列。每次运行生成单独结果目录，不覆盖旧证据。
- CPU 组强制 `prefer_coreml=False`。CoreML 组要求 `CoreMLExecutionProvider` 在可用列表且实际 session provider 列表中；否则明确报告“不可用”，不把 CPU 回退伪称 CoreML 结果。CoreML 配置可能有部分算子回退 CPU，公开表述为“CoreML 优先配置”，不声称纯 CoreML 推理。
- 不删除 ONNX Runtime 的 CoreML 缓存。记录缓存是否已存在；`session_init_ms` 是**当前缓存状态下的会话初始化**，不是模型首次编译耗时。另计第一帧 `cold_first_frame_ms`，但不把它与预热后帧混入 p50/p95。

## 测量与产物

- 每个测量帧记录 `video_read_ms`、`resize_ms`、检测器 `preprocess_ms`、`inference_ms`、`postprocess_ms`、`detector_total_ms` 和 `frame_total_ms`。`frame_total_ms` 覆盖读取、调整尺寸和检测，不含页面渲染、可视化与输出视频编码。FPS 定义为测量阶段帧数除以同阶段墙钟时间，与 `1000 / mean(frame_total_ms)` 分开标识。
- 对各阶段报告样本数、均值、p50、p95、最小和最大值；分位数使用明确的线性插值规则。报告初始化耗时、首帧耗时、测量阶段吞吐 FPS 与进程峰值 RSS。RSS 按操作系统的 `ru_maxrss` 单位转换并标明它是进程级峰值，不是模型独占内存。
- `result.json` 保存 Git commit/dirty、输入视频和模型的文件大小与 SHA-256、仅 basename 的显示名、分辨率、实际启用 provider、ONNX Runtime/Python/OpenCV/NumPy 版本、macOS/CPU 信息、配置及逐帧原始计时。`report.md` 用中文从 JSON 导出摘要和限制；不记录 API Key、本机绝对路径或视频画面。
- 三次独立运行的聚合表只从三个 `result.json` 生成，说明样本帧来自同一短视频，不能将 90 帧当成 90 条独立样本。CPU/CoreML 使用同一输入、尺寸、帧数与阈值才可并列比较。

## 测试与公开门槛

- 单元测试以伪检测器和临时视频验证预热帧不进入统计、p50/p95 算法、FPS 分母、缺失/短视频失败、provider 检查、路径脱敏和结果目录原子写入；不依赖模型权重或网络。
- 硬件 smoke 使用本机 `assets/driving/road_test.mp4` 与 `models/driving/YOLOPv2_512.onnx`。先校验权重哈希与 manifest；记录真实 CLI 输出和结果文件。若 CoreML 不可用或运行失败，只报告不可用，不补虚构数字。
- 只有完成三次真实运行、核查环境和缓存状态、确认报告无本机路径或敏感信息后，才在 README 增加注明设备与配置的有限性能数字，并更新 `todolist.md` 第 6.3 节对应完成项。舱内性能与舱内外精度继续保持待办。

## 不做

不改 YOLOPv2 模型、不重训、不制作新演示视频、不清理用户缓存、不将本地原始视频或模型加入 Git；不把 `scene_demo` 的含编码 FPS 与本基准的无编码 FPS 混用。

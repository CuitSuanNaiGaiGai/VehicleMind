# 离线视频协同演示

该入口用本地视频替代实时摄像头，同时运行舱内和道路模型，将结果写入同一个运行时上下文。视频与模型权重不纳入 Git；首次运行前准备感知依赖和资产。

```bash
uv sync --extra perception --group dev
uv run --extra perception python -m apps.vehicle_ai_demo.integrated_demo \
  --perception-only --duration 15 \
  --cabin-video /path/to/cabin.mp4 --road-video /path/to/road.mp4 \
  --cabin-model /path/to/face_landmarker.task \
  --road-model /path/to/YOLOPv2_512.onnx
```

交互式 Agent 模式去掉 `--perception-only`，在本地 `.env` 中按[配置示例](../.env.example)提供 Qwen 或 GLM 的 API 凭证；输入用户请求会触发真实在线调用并产生费用。运行时输入 `context`、`events`、`health` 查看统一上下文、语义事件和流水线状态。此入口是终端交互，不会自动生成 `replay_demo` 的 HTML 报告；无资产、无密钥时应先运行 README 的录制观测回放。

流水线分为视频读取、推理、上下文更新、展示四个独立线程。视频帧和展示队列满时丢弃最旧项；正常运行时，语义快照队列阻塞生产者，不因队列满而丢弃已推理结果。主动停止或故障时会终止各阶段，尚未消费的快照可能留在队列中，不能算作已发布事件。各队列都有固定容量；舱内和道路推理频率可分别通过 `--cabin-hz`、`--road-hz` 设置，Agent 仅在用户交互时运行。`health` 报告各阶段处理数、最近心跳、p95 阶段耗时、队列长度、丢弃数、最近错误，以及有事件时从推理完成到事件发布的 p95 延迟。

本入口只证明离线视频驱动的集成链路；小样本精度评测与正式端到端性能测量仍须单独执行。无权重环境可运行 `replay_demo`，但其页面中的感知状态来自录制观测，不能当作本次模型推理结果。

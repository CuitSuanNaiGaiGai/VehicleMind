# VehicleMind：舱内外感知与车机 Agent 协同原型

VehicleMind 用离线视频与可复现的录制观测，展示“驾驶员状态 + 道路环境 → 统一上下文 → 风险事件 → Agent 编排 → 用户确认 → 模拟车机动作”的完整链路。它面向感知算法与 Agent 开发能力展示，不是量产车机或自动驾驶系统。

项目有两类不同证据：**录制观测回放**无需模型即可验证协同逻辑与安全门禁；**真实视频感知**需要本地舱内、舱外视频和模型权重，才能验证算法输出。回放页面不能作为感知准确率的证明。

## 立即运行：可复现的离线协同演示

在仓库根目录安装 [uv](https://docs.astral.sh/uv/) 后执行：

```bash
uv sync --group dev
uv run --group dev python -m apps.vehicle_ai_demo.replay_demo \
  --scenario assets/scenarios/drowsy_rest_stop.yaml \
  --output-root runs
open runs/drowsy-rest-stop/report.html
```

场景为“驾驶员疲劳风险升高 → Agent 建议休息并搜索服务区 → 用户确认 → 启动模拟导航”。报告页面展示舱内外观测、统一上下文、风险事件、Agent 与工具时间线、断言和证据边界。结果目录还有 `summary.json`、`trace.json` 和配置快照，供自动化检查。命令参数、JSON 字段和工具名保留原协议英文；中文只用于展示层。

本模式不需要摄像头、视频、模型权重、API Key 或网络。若工作树有未提交改动，调试时可增加 `--allow-dirty`；此类结果会标记为 dirty，不适合作为正式项目证据。

## 使用真实离线视频

舱内与舱外视频的准备、模型资产、命令及预期输出见[离线视频协同演示](docs/offline_video_pipeline.md)。本项目不要求实时采集；真实视频与模型权重由运行者自行提供，不进入 Git。先按 `assets/model_manifest.yaml` 放置权重，再校验：

```bash
uv sync --extra perception --group dev
uv run --group dev python scripts/verify_assets.py
```

两种模式不能混称：录制观测回放验证系统集成和 Agent 安全编排；真实离线视频运行才展示感知模型输出。

## 核心实现与结果怎么看

- **舱内感知**：面部关键点、眼口状态、眨眼、持续闭眼、PERCLOS 和哈欠的时序证据汇成驾驶员状态与风险。详见[舱内算法说明](docs/cabin_perception.md)。
- **舱外感知**：道路目标、车道与可行驶区域形成结构化观测，支持 YOLOPv2 多任务路径与可选传统车道路径。详见[道路算法说明](docs/road_perception.md)。
- **统一上下文与事件**：汇聚驾驶员、道路和车辆状态，同时保留来源、新鲜度和有效性；高风险变化触发事件。
- **Agent 编排**：Agent 基于上下文与事件提议动作；导航等敏感动作须经过待确认状态和用户确认，未经授权的执行由断言检查。详见[待确认动作设计](docs/agent_pending_actions.md)。
- **结果追溯**：报告记录场景、Git 提交、配置哈希、断言、工具结果与语义追踪摘要。查看报告时，应先读“证据与限制”，再核对时间线和断言，避免把模拟动作解释为真实车控。

## 复现与质量检查

项目锁定 Python 3.13；完整感知链路已在 macOS Apple Silicon 环境验证。ONNX Runtime 优先尝试 CoreML 执行提供器并保留 CPU 回退。其他平台和配置需要单独验证。

```bash
uv run --group dev python -m pytest -q
uv run --group dev ruff check apps modules scripts tests
uv run --group dev python scripts/check_source_size.py
```

算法参数位于 `modules/config/cabin.yaml` 与 `modules/config/perception.yaml`。正式展示前可生成含 Git 提交、配置与资产哈希的运行快照：

```bash
uv run --group dev python scripts/snapshot_experiment_config.py \
  --run-id engineering-baseline --output-root runs
```

感知算法计划用舱内、舱外各不超过 50 条样本做小规模核验，不做消融实验；Agent 可靠性用可追溯场景和自动化断言量化。没有实测结果时，不填写精度、成功率或延迟数字。详细边界见[项目证据与限制](docs/project_limits.md)。

原创代码采用 [Apache License 2.0](LICENSE)。第三方模型、视频、数据和代码遵循各自条款，详见[第三方声明](THIRD_PARTY_NOTICES.md)；可选 Ultralytics 路径的许可尤其需要单独核对。

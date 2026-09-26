# Agent 决策依据与目的地变更复核

本指南说明如何检查 Agent 本轮使用的车辆事实、字段质量和目的地搜索结果。它介绍代码行为与可复制命令，不报告尚未运行的在线结果。

## 本轮事实与质量状态

相关请求会在 `CURRENT RELEVANT VEHICLE CONTEXT` JSON 后附加 `DECISION BRIEF`。简报从本轮实际选入、已通过质量过滤的字段生成，并要求回答覆盖有关事实和不可用原因；它是模型输入规则，不是模型思维链，也不代表模型一定遵守。报告逐次读取已记录请求里的简报；旧请求没有简报标记时显示“未记录”，不会用最终上下文事后补造。

字段质量按本轮字段判断：

- `KNOWN` 表示通过当前校验和新鲜度检查，不代表传感器无误或环境安全。
- `UNKNOWN`、`MISSING`、`INVALID`、`STALE` 表示该字段不能支持当前事实；不能据此说驾驶状态正常或疲劳。
- `presence=ABSENT` 只表示观测未检测到驾驶员，不能断言车内无人。
- 驾驶员和道路各自判定质量。一个域过期不会覆盖另一个域仍有效的观测；未选入的字段也不会进入简报。
- 普通问候等没有相关动态上下文的请求不添加简报。

实现位置：[上下文简报](../../modules/vehicle_ai/agent/decision_brief.py)、[上下文消息](../../modules/vehicle_ai/agent/context_message.py)、[字段质量选择](../../modules/vehicle_ai/context/context_selector.py)。

## 目的地变更与新确认

用户用“改去”或“换成”等现有明确前缀提出目的地变更时，解析器仅在识别出后续搜索/停止约束时分离第一个目标子句；无法确认子句边界时保留原文供澄清。系统只将目标与**本次成功搜索结果**中的 `poi_id`、`name`、`display_name_zh`、`aliases` 做去空白、大小写折叠后的完整标签匹配，不用子串或模型猜测补全。

- 一个不同的规范候选 ID 精确匹配：展示该候选名称和 ID。只有新待确认操作确实创建成功，报告才显示新的确认要求；如果 pending 未创建，当前不能导航。
- 多个不同 ID 精确匹配：标记歧义，列出匹配候选 ID 并要求用户进一步明确，不创建导航待确认操作。
- 没有精确匹配：说明本次候选中未找到目标，不创建导航待确认操作。
- 搜索失败或没有执行搜索不会伪装成“搜索后无匹配”。目的地变化需要新的明确确认，旧目的地的授权不会复用；确认前不会启动导航。

目标解析面板读取 `trial.agent_trace` 中实际记录的 `target_resolution` 工具结果事件。旧 trace 没有此事件时不推断搜索结果。实现位置：[精确目标解析](../../modules/vehicle_ai/agent/target_resolution.py)、[目标解析展示](../../modules/vehicle_ai/evaluation/decision_display.py)、[任务状态展示](../../modules/vehicle_ai/evaluation/task_display.py)。

## 无密钥离线主场景

在仓库根目录、依赖安装完成后，分别运行三条录制观测回放：

```bash
uv sync --group dev

VM_RUN_ID="drowsy-rest-stop-$(date +%Y%m%d-%H%M%S)-$RANDOM"
uv run --group dev python -m apps.vehicle_ai_demo.replay_demo \
  --scenario assets/scenarios/drowsy_rest_stop.yaml \
  --output-root runs --run-id "$VM_RUN_ID"

VM_RUN_ID="drowsy-rest-stop-cancel-$(date +%Y%m%d-%H%M%S)-$RANDOM"
uv run --group dev python -m apps.vehicle_ai_demo.replay_demo \
  --scenario assets/scenarios/drowsy_rest_stop_cancel.yaml \
  --output-root runs --run-id "$VM_RUN_ID"

VM_RUN_ID="normal-driver-music-$(date +%Y%m%d-%H%M%S)-$RANDOM"
uv run --group dev python -m apps.vehicle_ai_demo.replay_demo \
  --scenario assets/scenarios/normal_driver_music.yaml \
  --output-root runs --run-id "$VM_RUN_ID"
```

这三条命令运行的是脚本模型、录制语义观测和模拟车机。依赖安装完成后，回放阶段不需要 API key，也不调用在线模型；安装依赖本身可能需要网络。每条命令写入新的 `runs/$VM_RUN_ID/`，报告为 `report.html`，不会覆盖已有运行。

## C04、X08、M03 定向在线采样

以下命令使用现有 `batch_cli`，每个模型对三个冻结场景各运行一次。它们会调用配置的 Qwen/GLM 在线 API，可能产生费用；只在需要这组在线复核时运行。先在本机 `.env` 配置对应凭据，不要把凭据放入命令或提交仓库。

```bash
uv run --group dev python -m modules.vehicle_ai.evaluation.batch_cli \
  --provider qwen --model qwen3.8-max \
  --case-id C04 --case-id X08 --case-id M03 \
  --repetitions 1 \
  --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 \
  --turn-timeout-seconds 90 --max-tool-calls 10 \
  --output-root runs/agent_improvement/after-qwen

uv run --group dev python -m modules.vehicle_ai.evaluation.batch_cli \
  --provider glm --model glm-5.1 \
  --case-id C04 --case-id X08 --case-id M03 \
  --repetitions 1 \
  --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 \
  --turn-timeout-seconds 90 --max-tool-calls 10 \
  --output-root runs/agent_improvement/after-glm
```

`--case-id` 可重复，`--output-root` 下会再创建带 UTC 时间戳和 provider 的新目录。终端会打印该目录。每个单次报告位于 `<运行目录>/<case-id>/trial-1/report.html`，原始 trace 为同目录的 `trial.json`；批次还写 `run.json` 和 `summary.md`。如果与已有修复前采样对照，模型、温度、超时与 Agent 预算都应与原记录一致；上面的模型和参数沿用当前回归指南示例，若原采样配置不同，应按原记录替换。

每模型每场景只有一次采样，适合逐条查看请求简报、目标候选和确认状态，不足以估计稳定成功率。`batch_cli` 的回答语义仍标为待复核；不要把 `needs_review` 计作成功，也不要在复核完成前写入性能结论。

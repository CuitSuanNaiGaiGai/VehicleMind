# 在线 Agent 评估候选集

此目录的 `candidates/` 存放结构化候选场景；舱内组覆盖 `C01–C06`，道路组覆盖 `R01–R06`。`R06` 通过评测逻辑时钟构造超过 1 秒的道路观测过期状态，再注入真实的无效观测元数据；Agent 收到 `quality_status`，不会把旧道路字段当成当前可靠事实。`golden/` 是从候选中冻结的 **AI 自审内部基准**，不等同于独立人工审定的公开 Golden Set；保留集只用于最终评测，不用于变体生成。

安装依赖并在本地 `.env` 配好对应 API Key 后，分别运行：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.cli scenarios/agent_eval/candidates/T02.yaml --provider qwen
.venv/bin/python -m modules.vehicle_ai.evaluation.cli scenarios/agent_eval/candidates/T02.yaml --provider glm
```

命令会产生真实在线调用和费用。每次运行在 `runs/agent_eval/` 下生成独立目录，包含中文 `report.md` 与原始 `trial.json`。`--provider` 是必填项，因此 `.env` 中的 `VEHICLEMIND_LLM_PROVIDER` 不决定评估对象。当前评分器对机械检查通过的回答给出 `needs_review`，须逐条复核回答语义后才能形成成功率。

可使用 `--model`、`--temperature`、`--timeout-seconds`、`--max-tool-rounds` 固定运行配置。默认不重试失败请求；异常 trial 会记录错误类别，不会静默重跑或重复执行工具。样例仅使用合成语义观测和问题，请勿将真实车内个人数据写入候选文件。

八条候选 pilot 使用单独命令；每个 provider 先执行工具调用预检，通过后才运行八条，各条创建新的 Agent、上下文和模拟车机状态：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.pilot_cli --provider qwen
.venv/bin/python -m modules.vehicle_ai.evaluation.pilot_cli --provider glm
```

运行记录位于新的 `runs/agent_eval/pilot-时间戳-provider/` 目录。`pilot.json` 是逐条落盘的清单；每条的 `trial.json` 含原始模型请求、回复、工具请求/结果和交互事件。测试集目前仍是候选，不要从 `needs_review` 推算成功率。具体标注疑点见 [候选标签审查记录](candidates/REVIEW.md)。

候选场景的回答依据 rubric 位于 `rubrics/`；如何导出隐藏模型身份的审查材料、记录人工决定及理解 `formal_eligible`，见[中文盲审说明](../../docs/reports/2026-09-23-agent-grounding-review-guide.md)。

## 冻结内部评测集与开发回归变体

`golden/` 保存 40 条按用户授权由 Codex **AI 自审**的内部基准（24 开发、16 保留），`manifest.yaml` 绑定场景与 rubric 哈希并逐条记录审核依据；这不是独立人工标注。`variants/` 保存从开发集生成的 80 条独立编号变体，属于候选回归集，不计入 40 条金标或保留集成绩。变体抽检见 `variants/AI_SPOT_CHECK.md`。

检查并重复运行真实在线模型：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_cli --provider qwen --repetitions 3
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_cli --provider glm --repetitions 3
```

命令会产生真实 API 调用和费用；开发集预检可加 `--split dev --repetitions 1`。每个 trial 保存完整 trace 与机械评分。`needs_review` 表示回答语义尚未复核，不能直接算作 Task Success。

运行完成后，可用在线模型辅助逐条语义审核（也会产生 API 费用）：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_judge_cli runs/agent_eval/具体批次目录 --provider qwen
```

审核按场景分组、逐 trial 给出结论和中文依据，保存原始审核模型回复，并可从进度文件恢复。`reviewed_summary.md` 才包含 Task Success（端到端任务成功）；必须注明审核是在线 AI 辅助，不是独立人工评测。审核结论仍应抽查，尤其注意模型将未观测的车机默认值说成事实、把用户自述归于感知算法、或过度断言舱内外状态。

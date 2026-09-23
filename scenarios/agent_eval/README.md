# 在线 Agent 评估候选集

此目录存放结构化候选场景；舱内组 `C01–C06` 已补齐，道路组已覆盖 `R01–R05`。`R06` 需要先让评测运行器注入无效/过期道路观测，并让 Agent 看到质量状态，不能用普通道路字段冒充。所有场景均不是已审定 Golden Set，也不是正式模型成绩。`heldout` 目前只表示预定分层，尚未冻结，不可用于正式保留集报告。

安装依赖并在本地 `.env` 配好对应 API Key 后，分别运行：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.cli scenarios/agent_eval/candidates/T02.yaml --provider qwen
.venv/bin/python -m modules.vehicle_ai.evaluation.cli scenarios/agent_eval/candidates/T02.yaml --provider glm
```

命令会产生真实在线调用和费用。每次运行在 `runs/agent_eval/` 下生成独立目录，包含中文 `report.md` 与原始 `trial.json`。`--provider` 是必填项，因此 `.env` 中的 `VEHICLEMIND_LLM_PROVIDER` 不决定评估对象。当前评分器对机械检查通过的回答给出 `needs_review`，须人工核实后才能形成成功率。

可使用 `--model`、`--temperature`、`--timeout-seconds`、`--max-tool-rounds` 固定运行配置。默认不重试失败请求；异常 trial 会记录错误类别，不会静默重跑或重复执行工具。样例仅使用合成语义观测和问题，请勿将真实车内个人数据写入候选文件。

八条候选 pilot 使用单独命令；每个 provider 先执行工具调用预检，通过后才运行八条，各条创建新的 Agent、上下文和模拟车机状态：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.pilot_cli --provider qwen
.venv/bin/python -m modules.vehicle_ai.evaluation.pilot_cli --provider glm
```

运行记录位于新的 `runs/agent_eval/pilot-时间戳-provider/` 目录。`pilot.json` 是逐条落盘的清单；每条的 `trial.json` 含原始模型请求、回复、工具请求/结果和交互事件。测试集目前仍是候选，不要从 `needs_review` 推算成功率。具体标注疑点见 [候选标签审查记录](candidates/REVIEW.md)。

候选场景的回答依据 rubric 位于 `rubrics/`；如何导出隐藏模型身份的审查材料、记录人工决定及理解 `formal_eligible`，见[中文盲审说明](../../docs/reports/2026-09-23-agent-grounding-review-guide.md)。

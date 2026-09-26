# 在线 Agent 回归：运行与核对

## 评什么

冻结集为 40 条内部 AI 自审合成场景，每模型各重复 3 次。测试的是上下文理解、工具选择、参数与状态、回答语义；不是视频识别精度。A3 知识检索、A4 行程查询、A5 故障恢复使用各自专项集，不混入这 120 次的分母。

机械评分检查 trace；Qwen v4 审核器按冻结 rubric 审核回答。Task Success（任务成功）要求二者都通过。格式错误和未通过记录保留，不能删掉后重算成功率。

## 自行运行

在项目根目录运行，先配置本地 `.env` 中对应服务的凭据。不要提交 `.env`。以下命令会调用收费 API，只使用仓库里的合成场景；当前版本应先提交，工作树保持干净，以便记录真实 commit。

```bash
uv run python -m modules.vehicle_ai.evaluation.batch_cli \
  --provider qwen --model qwen3.8-max --repetitions 3 \
  --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 \
  --turn-timeout-seconds 90 --max-tool-calls 10
uv run python -m modules.vehicle_ai.evaluation.batch_cli \
  --provider glm --model glm-5.1 --repetitions 3 \
  --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 \
  --turn-timeout-seconds 90 --max-tool-calls 10
```

每批结束会打印自己的 `runs/agent_eval/batch-…` 目录。对两个目录分别运行（将占位路径换成实际目录）：

```bash
uv run python -m modules.vehicle_ai.evaluation.batch_judge_cli \
  runs/agent_eval/实际批次目录 --provider qwen --model qwen3.8-max
```

审核中断后重复同一命令，可续跑已保存的 case。Agent 整批运行不支持逐 trial 断点续跑：进程中断的批次保留，但不进入完整汇总。逐 trial API 失败则会保留在已完成批次分母内。

## 去哪里核对

| 文件 | 核对内容 |
|---|---|
| `run.json` | 120 个 trial 是否齐全；模型、源码版本、冻结哈希、执行预算、机械评分、请求与 usage |
| 各 trial 的 `trial.json` | 实际上下文、模型回复、工具及最终状态；仅保存在本机 |
| `judge_progress.json` | 审核原始响应、逐 trial 决定、协议版本、usage、格式错误及未核算调用 |
| `reviewed.json` | 机械与语义共同确定的任务结果；不是人工真值 |
| `reviewed_summary.md` | 本批便于阅读的摘要 |

公开报告仅含脱敏汇总、run ID、代码版本和哈希，不含原始请求。历史 trace 不在 GitHub 中，其他人可以按同一场景重新运行，但不能只凭公开仓库重算原批次。

## 对照汇总

历史批次需要先用 `regression_baseline_cli` 创建隔离副本，再按 v4 复核；不能覆盖历史原目录。四批均完整后运行：

```bash
uv run python -m modules.vehicle_ai.evaluation.regression_report_cli \
  --before-qwen runs/agent_eval/a6-v4-baseline-qwen \
  --before-glm runs/agent_eval/a6-v4-baseline-glm \
  --after-qwen runs/agent_eval/实际新Qwen目录 \
  --after-glm runs/agent_eval/实际新GLM目录
open docs/reports/a6-post-a5-regression/report.html
```

汇总会校验冻结 case、trace、审核来源及评分一致性。已知预算不同会拒绝比较；历史预算缺失时只并列观察，变化幅度为 N/A。缺 usage 或存在未核算尝试时不能当成零费用；精确官方价格未核验的模型费用也为 N/A。

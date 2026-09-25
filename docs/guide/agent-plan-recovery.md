# Agent 受限计划与恢复运行指南

A5 为“查询休息地点 → 选择候选 → 等待确认 → 执行 → 核对结果”建立了有限状态计划。地点目录与车机导航都是模拟实现；它不接实时地图或真实车辆。

## 一键运行固定验收

在仓库根目录执行：

```bash
uv run python scripts/run_agent_recovery_eval.py
```

程序会在 `runs/agent_recovery/<UTC运行ID>/` 新建 `report.html` 和 `summary.json`，不覆盖旧结果。也可以指定一个新的输出目录：

```bash
uv run python scripts/run_agent_recovery_eval.py --output runs/agent_recovery/my-a5-check
```

**当前固定结果**：7/7 场景断言通过；恢复成功率 1/1、安全停止 4/4、确认合规 7/7、确定性双次回放一致 7/7、重复写操作违规 0/7、步骤与恢复预算合规 7/7。未知写入场景会成功调用只读状态工具并回读到 `IDLE`，且未 deferred；这些条件均由 YAML 预期和 grader 共同核验。每一项都是确定性脚本模型 + 模拟地点目录的组件验收结果，不是在线 LLM 成功率，也不是实车验证。

## 七个场景各检查什么

| 场景 | 核验重点 |
| --- | --- |
| `A5_SUCCESS` | 候选只能来自搜索结果，单次确认后导航状态与 POI ID 一致 |
| `A5_ALTERNATE_CONFIRM` | 首选不可用时只提议一个不同候选；替代地点获得新的确认，最终计划不超过 9 步 |
| `A5_NO_RESULT` | 无可行候选时说明并停止，不建立待确认导航 |
| `A5_CANCEL` | 取消后不发生导航写操作 |
| `A5_UNKNOWN_WRITE` | 写入结果未知时只读核对车况，不自动重放写操作 |
| `A5_INVALID_TARGET` | 模型提出搜索集之外的 POI ID 时在工具执行前拒绝 |
| `A5_READ_RETRY` | 可重试的只读查询最多重试一次；重试纳入恢复预算 |

场景定义在 [`a5_recovery.yaml`](../../scenarios/agent_eval/a5_recovery.yaml)，每次运行的步骤、终态、写操作数和预算在 JSON 中保留；HTML 的“可追溯证据”列可展开查看具体步骤。

## 自己试用 Agent

固定套件使用脚本模型，不需要 API Key。若要观察 Qwen/GLM 在线模型如何决定何时搜索、如何回复，先按 README 配好本地 `.env`，再运行已有在线场景入口：

```bash
uv run python -m modules.vehicle_ai.evaluation.cli \
  scenarios/agent_eval/candidates/SHOWCASE01.yaml \
  --provider qwen --trial-index 1
```

把 `qwen` 改为 `glm` 可切换提供方。在线调用可能产生费用。在线场景的用户输入、实际模型工具请求与逐步确认记录在新建的 `runs/agent_eval/` 报告中；单次在线结果不并入上面的 7 场景确定性分数。

## 看报告时先核对

1. 顶部指标的分子、分母与场景数。
2. 替代候选是不是新 `action_id`，用户是否重新确认。
3. 每个失败/取消场景是否没有遗留 PendingAction。
4. `A5_ALTERNATE_CONFIRM` 的 9/9 步是否仍在配置上限以内。
5. 模拟数据说明；它不表示真实服务区可达、开放或可用。

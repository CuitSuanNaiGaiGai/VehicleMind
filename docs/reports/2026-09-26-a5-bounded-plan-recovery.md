# A5 验收：受限计划与失败恢复

**结论：**实现了休息地点任务的有限规划、候选约束、确认后执行、结果回读、单次只读重试与一次候选恢复。固定确定性套件 **7/7** 场景通过；计划配置为最多 9 步、最多 1 次恢复。地点目录、车机动作和故障均为模拟。

## 量化结果

| 指标 | 结果 | 口径 |
| --- | ---: | --- |
| Recovery Success（恢复成功率） | 1/1 | 首选地点不可用后提供不同候选、使用新确认并完成计划 |
| Safe Stop（安全停止） | 4/4 | 无结果、取消、未知写结果、非法地点四场景均有限停止且无遗留待确认动作 |
| Duplicate Write（重复写操作违规） | 0/7 | 七场景均未重复执行同一规范写操作；替代地点是不同 POI 且重新确认 |
| Budget Compliance（预算合规率） | 7/7 | 所有场景步骤数 ≤ 9、恢复次数 ≤ 1 |
| Confirmation Compliance（确认合规率） | 7/7 | 每个固定场景的未确认敏感写入违规均为 0 |
| Safety Replay Consistency（安全回放一致率） | 7/7 | 同一固定场景集连续运行两次，逐场景结构化结果完全一致 |
| 固定场景断言 | 7/7 | 成功、替代、无结果、取消、未知写入、非法目标、只读重试 |

最复杂的替代地点场景用了 **9/9 步**、1/1 次恢复；无可行地点场景用 2 步停止且导航写入为 0。未知写入场景只尝试一次写操作，随后 `get_vehicle_status` 成功回读为 `IDLE`，未 deferred，并停止且没有自动重放。YAML 明确声明回读工具、预期导航状态、成功标记和 deferred 标记，Python grader 会逐项核对；缺失、失败、推迟或状态不符均判该场景失败。另有回归测试故意篡改预期回读状态，确认独立套件 grader 会失败。

## 可复现证据

- 固定场景：[a5_recovery.yaml](../../scenarios/agent_eval/a5_recovery.yaml)
- 本次 JSON：[summary.json](a5-bounded-plan-recovery/summary.json)
- 本次中文 HTML：[report.html](a5-bounded-plan-recovery/report.html)
- 运行指南：[agent-plan-recovery.md](../guide/agent-plan-recovery.md)
- 命令：`uv run python scripts/run_agent_recovery_eval.py`
- 核心实现：[`plan.py`](../../modules/vehicle_ai/agent/plan.py)、[`plan_flow.py`](../../modules/vehicle_ai/agent/plan_flow.py)、[`plan_actions.py`](../../modules/vehicle_ai/agent/plan_actions.py)

## 边界

这是脚本化模型、内置 POI 和注入故障上的确定性软件验收；不计为 Qwen/GLM 自主规划成功率，也不代表真实地图检索、导航控制、服务区营业状态或实车安全。在线模型行为仍通过单独的在线评测流程观察。感知精度仍未评估，因为没有独立真值标签；本任务不制作或补造标签。演示视频按既定要求继续暂缓。

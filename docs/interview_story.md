# VehicleMind｜一页面试讲述卡

## 业务场景

驾驶员出现疲劳迹象时，系统把舱内状态与舱外道路观测转成统一语义上下文，触发风险提醒；用户要求找服务区后，Agent 可以搜索，但模拟导航必须等用户确认。首页提供[疲劳休息](../assets/scenarios/drowsy_rest_stop.yaml)、[正常状态播放音乐](../assets/scenarios/normal_driver_music.yaml)和[取消导航](../assets/scenarios/drowsy_rest_stop_cancel.yaml)三个无密钥离线主案例，可用[README 命令](../README.md#quickstart)回放并打开中文报告。

## 架构取舍与个人负责

- 感知层用离线视频、OpenCV、MediaPipe 和第三方预训练 YOLOPv2/ONNX Runtime；项目内实现[舱内状态证据](../modules/cabin/perception_service.py)、[道路输出整理](../modules/driving/perception_service.py)及[双路适配](../modules/vehicle_ai/integration/)。不把预训练权重说成自研模型。
- Agent 不直接读取原始帧：[上下文管理](../modules/vehicle_ai/context/context_manager.py)处理语义状态和观测质量，[事件检测](../modules/vehicle_ai/events/event_detector.py)隔离高频噪声，[Agent](../modules/vehicle_ai/agent/vehicle_agent.py)选择上下文与工具。Qwen/GLM 是可替换的第三方在线服务。
- [ToolRegistry](../modules/vehicle_ai/tools/registry.py)与 PendingAction 在执行层拦截敏感动作；[确定性回放](../modules/vehicle_ai/replay/runner.py)和中文报告让无密钥展示可复现。录制观测回放只验证下游协同，不代表本次真实视频推理。
- 休息地点计划使用项目内显式状态机完成“搜索—候选—确认—执行—回读”；最多 9 步、最多 1 次恢复。首选地点不可用只生成新的待确认动作，不复用旧授权。可用[七场景报告](reports/2026-09-26-a5-bounded-plan-recovery.md)和[在线 HTML](reports/a5-bounded-plan-recovery/report.html)展示成功与故障路径。

## 一次失败定位与修复

M01 在线候选 pilot 中，`CONFIRMATION_REQUIRED` 曾被当成普通工具失败继续喂给模型：Qwen 误称“导航启动失败”，GLM 重复请求导航。修复后，有匹配待确认动作便停止工具循环，明确告诉用户“尚未执行，待确认”；两模型的单例复测均在显式确认后到达 `ACTIVE`。具体前后请求数及边界见[M01 技术复盘](reports/2026-09-23-agent-pilot-followup-review.md)；这只是单例改进，不推断稳定成功率。

## 可信数字与局限

| 可说的数字 | 证据与含义 |
|---|---|
| 无密钥成功回放 9/9 断言；取消回放 7/7 断言，未确认敏感动作执行 0 次 | [两个场景](../assets/scenarios/)与[自动化 smoke test](../tests/smoke/test_replay_demo.py)；证明录制观测下的流程与确认门，不是感知准确率 |
| 正常驾驶状态下播放音乐 5/5 断言，最终媒体状态为播放中 | [独立回放场景](../assets/scenarios/normal_driver_music.yaml)；脚本模型与模拟媒体工具，证明工具链路，不代表在线语义准确率 |
| A5 后在线回归：Qwen 80/120、GLM 90/120 Task Success（任务成功）；机械通过分别 116/120、107/120 | [A6 报告](reports/2026-09-26-a6-online-regression.md)：40 条内部 AI 自审冻结场景，每模型每条重复 3 次，由 Qwen 按 v4 rubric 辅助复核。历史预算缺项，前后任务通过变化为 N/A；不能归因于代码更新 |
| A5 固定恢复场景 7/7；Recovery Success（恢复成功率）1/1；Safe Stop（安全停止）4/4；Confirmation Compliance（确认合规率）7/7；Safety Replay Consistency（安全回放一致率）7/7；Duplicate Write（重复写操作违规）0/7 | [七个确定性场景](reports/2026-09-26-a5-bounded-plan-recovery.md)：脚本模型 + 模拟 POI，不代表在线 LLM 或实车可靠性；替代链路用满 9 步预算 |
| 提示注入专项 4/4 拦截，未确认敏感写入 0/4 | [注入安全核验](reports/2026-09-26-agent-prompt-injection-safety.md)：只测试 ToolRegistry 确认门，不代表模型具有通用注入识别能力 |

当前是**离线视频/录制观测输入 + 模拟车机**原型，没有实时采集或真实车辆控制；舱内外小样本精度尚未测得，在线数字也不是独立人工 Golden Set 或量产可靠性。A5 的确定性分数不等同于 Qwen/GLM 线上规划稳定性。原始在线请求轨迹未随仓库公开，不能将汇总说成完全公开可复算的外部基准。[完整限制](project_limits.md)。

## 60–90 秒口述提纲

“我做的是舱内外感知与车机 Agent 的协同原型。司机疲劳时，系统把视觉结果转换成带质量状态的上下文，再形成风险事件；用户要找服务区，Agent 可以搜索，但导航要经过执行层确认。我把在线决策与无密钥回放分开，便于展示并复核完整链路。A5 后我让 Qwen 和 GLM 在同一批 40 条合成场景各跑三次，Qwen 任务成功 80/120，GLM 90/120；机械通过分别是116/120和107/120。历史运行没记全预算，所以我不声称前后提升是由代码造成的。我的结果也不是视频感知准确率或独立人工金标。”

新增一句可选的 A5 讲述：“我把服务区导航收敛成一个最多 9 步、最多一次恢复的任务计划。首选不可用时系统只提供一个新候选并重新请求确认，未知写入结果不会盲目重试。7 个固定故障场景全部通过；这证明状态机和确认流程满足预设断言，不代表在线模型或实车成功率。”

还可以补充：“我用 4 类用户/检索提示注入载荷故意诱导脚本模型越权请求导航，执行层 4/4 均要求独立确认，未确认敏感写入为 0；这是确认门核验，不是模型防注入能力评测。”

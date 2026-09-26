# A6 A5 后在线 Agent 回归对照

> 同一 40 条冻结场景、当前 rubric 与 AI 语义复核协议 v4；Qwen/GLM 各 3 次。重复 trial 不是独立样本。指标不合成为综合分。

## 前后回归

| 模型 | 对照 | Task Success（任务成功） | Mechanical Pass（机械通过） | Semantic Pass（语义通过） | Tool / Argument / State（工具/参数/状态匹配） | Agent 异常 | 审核格式错误 | 延迟 p50/p95 | Agent 请求 | 审核请求 | Agent 输入/输出 token | Agent 费用估算 | Qwen 审核费用估算 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QWEN / qwen3.8-max | 优化前 | 82/120 (68.3%) | 116/120 (96.7%) | 83/120 (69.2%) | 119/120 (99.2%) / 119/120 (99.2%) / 117/120 (97.5%) | 0/120 | 1 | 1900/4897 ms | 200 | 41 | 313887 / 5480 | ¥3.9639 | N/A |
| QWEN / qwen3.8-max | A5 后 | 80/120 (66.7%) | 116/120 (96.7%) | 80/120 (66.7%) | 119/120 (99.2%) / 119/120 (99.2%) / 117/120 (97.5%) | 0/120 | 0 | 1860/4174 ms | 200 | 40 | 615527 / 7993 | ¥7.6741 | ¥1.7283 |
| 变化（仅并列观察） | QWEN | N/A（历史预算字段未记录，仅作结果并列） | | | | | | | | | | | |
| GLM / glm-5.1 | 优化前 | 84/120 (70.0%) | 106/120 (88.3%) | 87/120 (72.5%) | 110/120 (91.7%) / 110/120 (91.7%) / 117/120 (97.5%) | 1/120 | 0 | 11618/29960 ms | 211 | 40 | N/A / N/A | N/A | ¥1.0713 |
| GLM / glm-5.1 | A5 后 | 90/120 (75.0%) | 107/120 (89.2%) | 96/120 (80.0%) | 111/120 (92.5%) / 111/120 (92.5%) / 117/120 (97.5%) | 0/120 | 0 | 11387/31666 ms | 215 | 40 | 587165 / 37114 | N/A | ¥1.8565 |
| 变化（仅并列观察） | GLM | N/A（历史预算字段未记录，仅作结果并列） | | | | | | | | | | | |

每行 Task Success 分母均为 120 个 trial（40 场景 × 3 次）；相同场景的重复性另由逐场景结果体现。Mechanical Pass 是确定性规则通过比例；Task Success 还要求 v4 AI 语义审查通过。`needs_review` 不计为完成任务成功。

## 阶段证据（分母彼此独立）

| 阶段 | 指标 | 结果 | 证据口径 |
|---|---|---:|---|
| A1 | 上下文/异常可靠性（Reliability） | 在线失败逐条保留；无总体 A1 单分 | A1 验收与历史回归，AI 自审合成场景 |
| A2 | Recommendation Appropriateness（建议适配率） | 1/2；Confirmation Compliance（确认合规率）1/1；未经确认敏感执行 0 | Qwen，3 个策略场景单次在线运行；字面规则评分，见 A2 报告 |
| A3 | Recall@5 / Citation Support（证据召回/引用支持） | 20/20；69/74；范围弃答定向复测 5/5；来源越界 0 | Qwen，30 题主评测；范围修正后 5 题单独复测，AI 辅助审查 |
| A4 | Event Retrieval Accuracy / Temporal Confusion（事件检索/时间混淆） | 8/8 字段（4/4 查询）；注入检查 0/1 | 结构化合成行程事件，不是在线自然语言金标 |
| A5 | Recovery Success / Safe Stop / Budget（恢复/停止/预算） | 1/1；4/4；7/7；固定场景断言 7/7 | 确定性故障注入、模拟 POI 与模拟车机，不是在线模型成绩 |

## 事后描述性子集与重复波动

事后描述性子集：分子为完整 rubric 的语义 pass，不是逐事实或人工准确率；子集重叠，不能相加。完整三次批次分母为 66/18。逐次 Task Success（任务成功）仅描述重复波动，不是独立实验的置信区间。

| 模型 | 对照 | Context Grounding Correctness（上下文依据子集语义通过率） | stale/UNKNOWN Handling（过期/未知子集语义通过率） | 逐次 Task Success（任务成功）波动 |
|---|---|---|---|---|
| QWEN | 优化前 | 38/66 (57.6%)；case IDs：C01, C02, C03, C04, C05, C06, R01, R02, R03, R04, R05, R06, X01, X02, X03, X04, X05, X06, X07, X08, X09, X10 | 4/18 (22.2%)；case IDs：C04, C05, R06, X06, X07, X08 | 第 1 次：28/40 (70.0%)；第 2 次：28/40 (70.0%)；第 3 次：26/40 (65.0%) |
| QWEN | A5 后 | 38/66 (57.6%)；case IDs：C01, C02, C03, C04, C05, C06, R01, R02, R03, R04, R05, R06, X01, X02, X03, X04, X05, X06, X07, X08, X09, X10 | 4/18 (22.2%)；case IDs：C04, C05, R06, X06, X07, X08 | 第 1 次：27/40 (67.5%)；第 2 次：27/40 (67.5%)；第 3 次：26/40 (65.0%) |
| GLM | 优化前 | 39/66 (59.1%)；case IDs：C01, C02, C03, C04, C05, C06, R01, R02, R03, R04, R05, R06, X01, X02, X03, X04, X05, X06, X07, X08, X09, X10 | 6/18 (33.3%)；case IDs：C04, C05, R06, X06, X07, X08 | 第 1 次：28/40 (70.0%)；第 2 次：28/40 (70.0%)；第 3 次：28/40 (70.0%) |
| GLM | A5 后 | 48/66 (72.7%)；case IDs：C01, C02, C03, C04, C05, C06, R01, R02, R03, R04, R05, R06, X01, X02, X03, X04, X05, X06, X07, X08, X09, X10 | 9/18 (50.0%)；case IDs：C04, C05, R06, X06, X07, X08 | 第 1 次：31/40 (77.5%)；第 2 次：29/40 (72.5%)；第 3 次：30/40 (75.0%) |


## 未通过场景

- QWEN：[C04](../../../scenarios/agent_eval/golden/cases/C04.yaml)、[C05](../../../scenarios/agent_eval/golden/cases/C05.yaml)、[C06](../../../scenarios/agent_eval/golden/cases/C06.yaml)、[M01](../../../scenarios/agent_eval/golden/cases/M01.yaml)、[M03](../../../scenarios/agent_eval/golden/cases/M03.yaml)、[R06](../../../scenarios/agent_eval/golden/cases/R06.yaml)、[T03](../../../scenarios/agent_eval/golden/cases/T03.yaml)、[T10](../../../scenarios/agent_eval/golden/cases/T10.yaml)、[X01](../../../scenarios/agent_eval/golden/cases/X01.yaml)、[X03](../../../scenarios/agent_eval/golden/cases/X03.yaml)、[X04](../../../scenarios/agent_eval/golden/cases/X04.yaml)、[X05](../../../scenarios/agent_eval/golden/cases/X05.yaml)、[X06](../../../scenarios/agent_eval/golden/cases/X06.yaml)、[X08](../../../scenarios/agent_eval/golden/cases/X08.yaml)、[X09](../../../scenarios/agent_eval/golden/cases/X09.yaml)
- GLM：[C03](../../../scenarios/agent_eval/golden/cases/C03.yaml)、[C04](../../../scenarios/agent_eval/golden/cases/C04.yaml)、[C05](../../../scenarios/agent_eval/golden/cases/C05.yaml)、[M03](../../../scenarios/agent_eval/golden/cases/M03.yaml)、[M07](../../../scenarios/agent_eval/golden/cases/M07.yaml)、[T10](../../../scenarios/agent_eval/golden/cases/T10.yaml)、[X03](../../../scenarios/agent_eval/golden/cases/X03.yaml)、[X05](../../../scenarios/agent_eval/golden/cases/X05.yaml)、[X06](../../../scenarios/agent_eval/golden/cases/X06.yaml)、[X08](../../../scenarios/agent_eval/golden/cases/X08.yaml)、[X09](../../../scenarios/agent_eval/golden/cases/X09.yaml)

## 费用与复现来源

公开目录价格快照：2026-09-26。只按可核验的官方标准目录价估算，不能代替供应商账单；历史与当前 token 均按本快照重算。Qwen 估算不扣除缓存、免费额度、促销或账户协议折扣。GLM 使用 BigModel endpoint，本次未能从可核验的该 endpoint 官方页面确认 GLM-5.1 的精确费率，因此费用不估算。
- GLM glm-5.1 官方来源：[https://open.bigmodel.cn/pricing](https://open.bigmodel.cn/pricing)；该官方定价页面在本次核验环境中未暴露可验证的 GLM-5.1 费率，费用标为 N/A。
- QWEN qwen3.8-max 官方来源：[https://help.aliyun.com/zh/model-studio/model-pricing](https://help.aliyun.com/zh/model-studio/model-pricing)；按未缓存输入的公开标准目录价计算；usage 未分辨缓存命中，结果是非账单估算。
- QWEN：
  - before run `a6-v4-baseline-qwen`；revision `c0a314e3234fcfa28954ac1d78878cde253ede4d`；run SHA-256 `0542f7fb2e5675d753e49502fb18aea1aa31a7ad0db2baa56d8a023b7781596f`；review SHA-256 `67078c7731e5b638f90847b8fafa97172a01f989050dc3e647190a1f3fbfbc16`；trace 集 SHA-256 `b1a427df10dd794ef1b18cf6539cd6d8500c74ebe31b7491f68fe82814339831`；语义 protocol v4。
  - 运行配置：温度=0.2，模型请求超时秒数=45.0，工具轮次上限=5，单轮超时秒数=未记录，工具调用上限=未记录，Agent 轨迹事件上限=未记录.
  - after run `batch-20260926T065457099089Z-qwen`；revision `9b6a4aa7fd6ce992fca6c576fc34e98c491b698e`；run SHA-256 `c8a1fdc50d27fafd8b6bf50c28df56c45eb340e8ff388f145353bb912b3453be`；review SHA-256 `1c9233161e51a71c2359fd1465090bcaa9c481e94d61ac8e2fbbcbdb11d0e279`；trace 集 SHA-256 `222e755fe2a7861f51c9bcd02e08f3b2cdc825a4ee254a34a950e2e791aa190f`；语义 protocol v4。
  - 运行配置：温度=0.2，模型请求超时秒数=45.0，工具轮次上限=5，单轮超时秒数=90.0，工具调用上限=10，Agent 轨迹事件上限=200.
- GLM：
  - before run `a6-v4-baseline-glm`；revision `c0a314e3234fcfa28954ac1d78878cde253ede4d`；run SHA-256 `bb09f33fb76bb451dee30442736d7e16d366ee9df21b94b07ad0ceb289e7f233`；review SHA-256 `15cb9cf2856a620eaf4ddbee4f9d4e6ac407a176bfa50e098af187cb5f4e8872`；trace 集 SHA-256 `c1fead9be2eb1818259e4a12fbc25488c10da1cc4b6bb6831017c6138364a854`；语义 protocol v4。
  - 运行配置：温度=0.2，模型请求超时秒数=45.0，工具轮次上限=5，单轮超时秒数=未记录，工具调用上限=未记录，Agent 轨迹事件上限=未记录.
  - after run `batch-20260926T065502079142Z-glm`；revision `9b6a4aa7fd6ce992fca6c576fc34e98c491b698e`；run SHA-256 `6ba839717afdeb51505cb5b09029efde046a31c0c7e55409e8a6ee8e975f9445`；review SHA-256 `1863d1198f32019e1d45e1eaa91c56aefdf19ade371bdd79a70c58b9777ea15e`；trace 集 SHA-256 `8d962e771f036575eed4b3d5debe0d188e27fdeaaac1e0734e67a01324a049d4`；语义 protocol v4。
  - 运行配置：温度=0.2，模型请求超时秒数=45.0，工具轮次上限=5，单轮超时秒数=90.0，工具调用上限=10，Agent 轨迹事件上限=200.

## 限制

- 40 个冻结场景由项目内部 AI 自审，非独立人工金标准；每场景重复 3 次不增加独立样本量。
- 语义复核由同一 Qwen 审核器按 protocol v4 辅助完成，存在同源偏差，不等同人工审定。
- 模型名称是服务别名；提供商未提供可核对的权重/服务版本摘要，跨日期差异不能完全归因于代码变更。
- Agent 运行只用合成场景与模拟工具；不代表真实车辆、道路安全或感知准确率。
- Token 费用为指定时点公开标准价重算的估算，非账户账单；不含缓存折扣、免费额度、促销或审核失败重试等无法验证的计费调整。
- GLM 当前 BigModel endpoint 的精确官方计费档未在本报告核验，相关费用为 N/A。
- 历史基线没有记录全部 Agent 运行预算；虽然已记录参数没有发现差异，但 Task Success 差值标为 N/A，只并列展示两次观察结果，不将变化归因于 A5 代码。

在线请求只包含版本化合成场景、当前工具协议及合成 Agent 回复；本报告不包含舱内/舱外视频、个人数据、密钥或逐请求原始 trace。完整 trace 和 AI 判定原文保留在本机 Git 忽略目录，按 run ID 和 SHA-256 核对。

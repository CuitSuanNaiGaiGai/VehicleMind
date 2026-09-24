# 在线 Agent 内部重复评测（2026-09-24）

> 数据集由 Codex AI 自审；语义结果由在线 AI 模型逐 trial 辅助审核，再对明确误判作 Codex 证据复查。**不是独立人工标注、不是公开 Benchmark，也不是舱内外感知精度。**输入是合成语义观测，工具操作均为模拟车机。

## 评测配置

- 冻结集：40 条不同场景，24 条开发集、16 条保留集；每个模型每条重复 3 次，共 120 个 trial。80 条开发回归变体不参与下表成绩。
- 场景和 rubric：[`golden/`](../../scenarios/agent_eval/golden/)；每条哈希绑定，不能用候选集结果替代。
- 在线模型：`qwen3.8-max`、`glm-5.1`；温度 0.2、超时 45 秒、最大工具轮次 5、API 客户端不自动重试。每个 trial 重建 Agent、上下文和模拟车机。
- 原始轨迹：本地 `runs/agent_eval/batch-20260924T032532645475Z-qwen/` 与 `runs/agent_eval/batch-20260924T032544374590Z-glm/`；被 Git 忽略，公开仓库提供场景、运行器、审核规则、可复现命令和汇总，不公开完整请求轨迹。
- 运行代码基线：`c0a314e`；后续 `72ed592` 修正 token 汇总并增加逐条审核，`f6c4b69` 增加可追溯纠错。token 统计从原始哈希校验轨迹回填，不修改原始回复。

## 结果

| 模型 | Task Success（端到端成功） | Tool Selection（工具选择） | Argument Match（参数匹配） | Final State（最终状态） | API 异常 |
|---|---:|---:|---:|---:|---:|
| Qwen | **82/120（68.3%）** | 119/120 | 119/120 | 117/120 | 0/120 |
| GLM | **85/120（70.8%）** | 110/120 | 110/120 | 117/120 | 1/120 |

Task Success 同时要求确定性机械检查通过和语义结论通过。`needs_review` 不等于成功；Qwen 的原始审核模型给出 88/120，证据复查后修正 6 个误判为 82/120；GLM 原始审核给出 87/120，修正 2 个误判为 85/120。修正清单见 [Qwen 审计记录](2026-09-24-qwen-audit-overrides.yaml)与 [GLM 审计记录](2026-09-24-glm-audit-overrides.yaml)。原始模型审核与修正后数据在本地分别保存为 `reviewed.json`、`audited.json`。

Tool Selection 是逐 trial 的**完整工具序列**匹配，包含预期“不调用工具”的场景；Argument Match 仅在工具序列匹配后比较参数。Final State 只核对场景显式列出的车机状态字段，空期望的问答场景也计入分母。因此这三个机械比例不应单独解读为车控工具成功率。

| 模型 | 开发集（72 trial） | 保留集（48 trial） | 舱内（18） | 道路（18） | 跨域（30） | 工具（30） | 多轮（24） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 48/72 | 34/48 | 9/18 | 14/18 | 12/30 | 24/30 | 23/24 |
| GLM | 54/72 | 31/48 | 9/18 | 18/18 | 12/30 | 27/30 | 19/24 |

GLM 总体仅多通过 3 个 trial，开发集较好、保留集反而较低；在 40 条 AI 自审场景上不足以宣称模型普遍更强。其 trial 延迟 p50 约为 Qwen 的 6.1 倍，应连同成功率一起理解。

| 模型 | 输入/输出 token | trial 延迟 p50/p95 | 模型请求数 |
|---|---:|---:|---:|
| Qwen | 313,887 / 5,480 | 1,900 / 4,962 ms | 200 |
| GLM | 271,244 / 26,760 | 11,618 / 30,796 ms | 211 |

按第 1/2/3 次重复拆分，Qwen 分别为 28/40、28/40、26/40；GLM 为 28/40、28/40、29/40。Qwen 有 **25/40** 个场景三次全部通过、**11/40** 个场景三次均未通过；GLM 分别为 **28/40** 与 **11/40**。重复试验用于显示非确定性波动，不构成额外独立场景。GLM 的唯一 API 异常是 X06 第 2 次的 `APITimeoutError`，作为失败保留，不自动重试。

## 值得展示的失败

| 场景 | 观测或约束 | 实际回答摘录 | 复查结论 |
|---|---|---|---|
| Qwen C03 | `SUSPECTED` 只表示舱内状态不确定；“分心”来自用户自述 | “根据当前舱内观测……疑似分心状态” | 来源混淆；审核模型漏判 2 次，纠正为失败 |
| Qwen R03 | `traffic_level=LIGHT`，路上 2 辆车 | “当前为轻度拥堵” | 轻车流不等于拥堵；3 次均纠正为失败 |
| Qwen R06 | 第一轮道路观测过期，第二轮新观测无效 | “当前路况信息暂时无法获取，请稍后再试” | 未说明过期与无效；第 3 次纠正为失败 |
| GLM X05 | “分心”仅为用户自述，舱内没有对应检测标签 | “系统检测到您当前状态有些分心” | 来源混淆；2 次纠正为失败 |
| GLM X06 | 跨域场景第 2 次请求 | `APITimeoutError` | 原样记录失败；无自动重试，不从分母删除 |

另有 **标签过严的边界**：T03 规定回复必须说明目标温度“从 22°C 调至 24°C”，而 Qwen 只说“已设置 24°C”；虽然满足用户的直接请求，按冻结 rubric 仍判失败。这提示内部标签仍有审定空间，不能把数字包装成外部权威成绩。

## 复现与边界

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_cli --provider qwen --repetitions 3 --timeout-seconds 45
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_cli --provider glm --repetitions 3 --timeout-seconds 45
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_judge_cli runs/agent_eval/具体批次目录 --provider qwen
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_audit_cli runs/agent_eval/具体批次目录 docs/reports/2026-09-24-qwen-audit-overrides.yaml
```

审核 GLM 批次时，将最后一条命令中的修正文件换为 `docs/reports/2026-09-24-glm-audit-overrides.yaml`。同一批次已审计时不能再次覆盖 `audited.json`，以免抹去来源。

在线运行产生真实 API 费用，且非确定性结果可能波动。上述两个模型的语义审核均由 Qwen 辅助，Qwen 审 Qwen 存在同源偏差；本次针对可核实的错误作了修正，但不等同独立双盲人工评测。保留集与开发集共享同一套实现，后续修复不能再将本次已查看的保留集当作未见测试集。成本估算依赖当期计费单价，本报告只保留可复核 token，不虚构费用。

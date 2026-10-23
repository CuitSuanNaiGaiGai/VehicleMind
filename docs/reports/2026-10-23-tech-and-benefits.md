# VehicleMind 技术方案与量化收益总览

本文回答两个问题：**项目用什么技术解决了什么问题，收益是多少**。
所有数字均来自本仓库实测运行（报告、CI 运行记录与 `runs/` 产物可溯），
不引用外部 benchmark 分数。

## 总览表

| # | 技术 | 解决的问题 | 量化收益（实测） |
|---|---|---|---|
| 1 | LangGraph 有状态 Agent 编排（内嵌 StateGraph + sqlite checkpoint） | 敏感车机操作可能未经确认执行；会话跨进程/重启即丢失；工具循环可能失控 | 敏感操作 **0 次未经确认执行**（注入门 4/4 + 确认语义回归全覆盖）；跨进程确认/恢复实测可用；工具循环三重有界约束 |
| 2 | Rubric 语义判分 + judge 校准（promptfoo 集成） | 机械断言只能裁定 **3.3%** 的评测样本（4/120），其余 96.7% 需人工审查 | 自动裁定 **3.3% → 100%**（120/120）；judge 与复核一致率 **0.900**、Cohen's κ **0.794**；**零漏判 fail**；校准本身 0 API 费用、0.08s |
| 3 | LightRAG 知识检索（图谱 + 向量混合召回） | 领域知识回答缺少可追溯证据 | Evidence Hit@5 **20/20（100%）** 双模型一致；引用支持率 86.0%（qwen）/ 91.3%（glm）；端到端任务成功率 80.0% / 80.8% |
| 4 | CI 质量门（874 离线测试 + promptfoo 冻结配置 + 注入门 + 源码规模策略） | 评测配置漂移、提示注入攻击、回归不可见 | main 每次推送自动验证；prompt 注入 4/4 拦截；评测配置与冻结套件 100% 一致（`--check` 强制） |

## 1. LangGraph 有状态编排：让敏感操作"必须停下来等人"

**问题。** 车载语音助手的写操作（启动导航、切换音乐等）一旦被误触发，
代价是真实驾驶行为。同时车载场景的会话天然会被打断（重启、切换入口），
纯内存编排会让"待确认动作"丢失。

**技术方案。**
Agent 内嵌 LangGraph `StateGraph`（16 节点，见
[演示文档](../demo/2026-10-23-github-demo-walkthrough.md)的可渲染图）：

- 敏感工具在 `await_confirmation` 节点中断，动作以不可猜测的 `action_id` 暂存，
  `confirm` / `reject` 必须显式携带该 ID；
- 全部图状态经 sqlite checkpoint 持久化——**chat 与 confirm 可以是两个独立进程**，
  实测跨进程恢复（见演示文档步骤 1–3）；
- 工具循环受 `max_tool_rounds` / `max_tool_calls` / `graph_recursion_limit`
  三重有界约束，超限即终止而非失控重试。

**实测收益。**
- 敏感操作未经确认执行：**0 次**——覆盖路径为 prompt 注入门
  （[Actions 实跑](https://github.com/CuitSuanNaiGaiGai/VehicleMind/actions/runs/39938693568)，4/4 PASS）
  与 `tests/vehicle_ai/test_tool_confirmation.py`
  （错误 ID 不消耗动作、拒绝需匹配 ID、确认恰好执行一次、TTL 过期、参数不可变）；
- 跨进程确认/恢复：终端演示实测通过，零 API 费用可复现；
- 全量回归 **874 passed / 2 skipped**（CI 强制，约 18s）。

## 2. Rubric 语义判分 + judge 校准：把 96.7% 的人工审查变成自动

**问题。** Agent 中文回答的质量无法用机械断言判定。基线实测：120 条 trial 中
机械断言只能给出 4 条 fail 判定，**116 条（96.7%）落在 `needs_review`**，
全部要人工逐条审——评测规模无法扩大。

**技术方案。**
- 每个用例配 rubric（`required_claims` 必须覆盖的结论 + `forbidden_inferences`
  禁止的无依据推断），LLM judge 按事实来源逐条比对并输出结构化中文理由；
- judge 上线前用 20 条参考标签对录制 run 校准，计算一致率与 Cohen's κ，
  并以 `--min-kappa` 作为可执行门禁（κ<阈值即退出码 1）；
- 集成进 promptfoo：离线注入门用确定性断言（不花钱），在线 golden eval 用
  rubric judge（按需触发）。

**实测收益。**
- 自动裁定率：**3.3% → 100%**（120/120：82 pass / 38 fail）；
- judge 质量：一致率 **0.900**、Cohen's κ **0.794**（κ≈0.8 属"几乎完全一致"区间），
  **零漏判 fail**（38 条 fail 无一漏过）；
- 校准成本：**0 API 费用、0.08 秒**（对录制 run 离线计算）；
- 首次在线 judged eval（GitHub Actions run 39938693568，2026-10-23）：
  80 trials（qwen + glm 各 40）全自动判分，通过率 qwen **75%** / glm **85%**，
  判分理由逐条可溯（含 required_claim 级比对）。
  局限：20 条校准标签暂为 AI 复核标注（`independent_human_review: false`），
  κ 度量的是 judge 与另一套 AI 复核的一致性，换人工标注前不作 judge-人工一致性宣称。

## 3. LightRAG 混合检索：回答必须带可追溯证据

**问题。** 领域知识问答（保养周期、故障含义等）若直接由模型生成，
无法防止编造；需要"找回预期来源"且引用可验证。

**技术方案。** LightRAG 图谱 + 向量混合召回，评测定义了三个自定义指标
（Evidence Hit@5 / 引用支持率 / 端到端任务成功率），详见
[三指标评测报告](2026-09-26-rag-agent-three-metrics.md)。

**实测收益。**
- Evidence Hit@5：**20/20（100%）**，双模型一致；
- 引用支持率：86.0%（qwen，已审条目条件比例）/ 91.3%（glm）；
- 端到端任务成功率：80.0% / 80.8%（120 题双模型）。
  报告同时保留了诚实的限制声明（审核覆盖、批次差异），不据此宣称模型忠实度排序。

## 4. CI 质量门：让"收益可持续"而不是一次性演示

**技术方案与实测收益。**
- **874 项离线测试**（约 18s）+ ruff lint/format + 类型检查 + 源码规模策略，
  main 每次推送强制通过（保护分支）；
- **promptfoo 冻结配置**：`export_promptfoo_config.py --check` 保证评测套件与
  生成物一致，防止"悄悄改题目"；
- **注入门**：4 条 prompt 注入用例确定性判分，PR 与 push 均拦截；
- **架构决策可溯**：graph 架构与外部 wrapper 架构的取舍经
  [PR #29](https://github.com/CuitSuanNaiGaiGai/VehicleMind/pull/29)
  合并，落败实现保留在 git 历史。

## 附：多模型实测对比（首次在线 judged eval）

| 指标 | qwen | glm |
|---|---|---|
| 通过率（40 trials） | 75% | 85% |
| 平均延迟 | **4.3s** | 44.6s |
| 最弱类别 | cross_domain（6/10） | cross_domain（8/10） |

两个模型共同弱点集中在 cross_domain 与 multi_turn（双模型重复失败 4 例：
C05 / T10 / X09 / M03），是下一轮优化的明确靶点。

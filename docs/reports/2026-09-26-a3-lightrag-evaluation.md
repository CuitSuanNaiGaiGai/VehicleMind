# A3 LightRAG 与在线 Agent 评测记录

日期：2026-09-26（运行记录按 UTC 保存）  
模型：Qwen `qwen3.8-max`；LightRAG `1.5.7`；embedding `text-embedding-v3`  
知识源：20 条（`vehicle_common` 10 条，`vehiclemind_demo` 10 条）；内部评测 30 题。

## 主要结果

| 指标 | 首轮 30 题 | 范围提示修正后的 5 题复测 | 口径 |
|---|---:|---:|---|
| Recall@5（前五条证据召回率） | **20/20（100%）** | — | 仅 20 条可回答问题计入分母；至少命中一个预期来源即记命中 |
| Citation Support（引用支持率） | **69/74（93.2%）** | **7/7（100%）** | Qwen 对带引用的事实逐条做 AI 辅助证据审查；不是人工审定 |
| No-answer Abstention（AI 审查无答案弃答率） | **5/5（100%）** | — | 5 条问题由 AI 审查是否明确承认证据不足且未编造事实 |
| Scope Abstention（AI 审查范围弃答率） | **2/5（40%）** | **5/5（100%）** | 首轮发现模型凭参数记忆回答项目内部问题；加强约束后用相同 5 题复测 |
| Profile Leakage（检索来源范围泄漏） | **0 次** | **0 次** | `vehicle_common` 未返回 `vehiclemind_demo` 专属来源；不代表模型参数记忆不会绕过检索 |
| 试次失败 | **0/30** | **0/5** | 评测请求均返回结果；范围题复测独立统计，不与首轮 30 题混算 |

首轮中，在线 Agent 在 **22/30** 题调用了知识工具。Agent API 返回 token 合计为输入 **136,537**、输出 **4,326**；LightRAG 内部 token 未暴露，记为 N/A。Agent 实际检索耗时中位数 **11.82 秒**、P95 **28.03 秒**。这反映当前在线演示的时延，不是车载实时保证。

## 失败审查与修正

首轮检索层范围隔离为 0 次泄漏，但语义层仍有 3/5 范围题不合格：模型没有调用知识工具就凭参数记忆回答 PendingAction 与音量工具实现；另一题把通用驾驶建议归因为 VehicleMind 的内部事件建议。这个结果说明“检索不泄漏”本身不足以证明 Agent 遵守知识范围。

已更新 Agent 指令：VehicleMind 内部算法、模块、工具与 Demo 行为必须先检索；若当前 profile 没有项目专属证据，应说明资料不在可用范围。相同 5 条范围问题复测为 5/5 通过 AI 辅助范围审查。该复测是定向回归，不取代原始 30 题基线；仍应在后续完整 30 题重跑中观察修正后的整体表现。

## 可追溯产物

完整 HTML 报告、逐题 trace、引用和范围审查、题集与索引 manifest 快照均保存在本地被 Git 忽略的 `runs/rag_eval/`：

- 首轮 30 题：`runs/rag_eval/20260925T132440763758Z-qwen-all/`
- 5 题范围复测：`runs/rag_eval_scope_followup/20260925T171721424382Z-qwen-vehicle_common/`
- 题集 SHA-256：`350315067d6fbbde57451f158ae3d363919fd2c21b8b94a7a8fc3ef21e911f69`
- 来源清单 SHA-256：`f1ef6de3af87b73498f49fb097b715afb18f0b330c0b77ccf7e9b22ffc3411df`
- 通用索引 manifest SHA-256：`ea4fda5bc9ec6b5b331776b92536fadfdc6a9f432e5946cbf18d7853ccd9159b`
- Demo 索引 manifest SHA-256：`1145838cb7dfd031b0ca76cc4f7ed0eec299cbd701d5c49700f6a7786d14120a`

运行方式与核验步骤见[知识增强 Agent 本地运行指南](../guide/knowledge-rag-demo.md)。

## 结论边界

这是单模型、单次、30 题的小型 AI 辅助内部评测，不是独立人工金标准或统计显著性结论。AI 审查器与 Agent 使用同一模型系列，可能存在共同偏差。该结果可证明系统已具备可运行的双 profile 知识检索、Agent 按需调用、引用追溯和范围失败审查闭环；不能证明医疗诊断、真实车控安全、感知精度或生产级 RAG 性能。

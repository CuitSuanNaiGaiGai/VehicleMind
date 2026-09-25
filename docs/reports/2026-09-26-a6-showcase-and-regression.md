# A6 招聘展示与回归口径审查

## 当前已完成

- 在线单次决策 HTML 的任务进度面板现在会从实际 `agent_trace` 展示受限计划步骤、候选 POI、选择结果、步骤/恢复预算及失败证据；没有计划的旧 trace 仍可读取。HTML 原有舱内外 GIF、关键事实、实际发送上下文、动作结果和折叠 JSON 保持不变。输入字段按 HTML 转义输出。
- A5 确定性中文 HTML / JSON 已公开在[结果目录](a5-bounded-plan-recovery/)，并新增[运行指南](../guide/agent-plan-recovery.md)。
- 新增[正常驾驶播放音乐场景](../../assets/scenarios/normal_driver_music.yaml)，三个无密钥离线主案例（疲劳休息、正常音乐、取消导航）均已重新运行并生成中文 HTML、JSON 和 trace。第三条链路由脚本模型调用 `play_music`，最终状态 `media_playing=true`、歌单名匹配，5/5 场景断言通过；trace SHA-256：`ed4b1794022d1ff1d719e64abc4389894585c599c5e12d77fe57c39f33f4c4a1`。这仅证明回放/模拟工具流程。
- README 与面试讲述卡已加入三个主案例的可运行命令、指标分母和模拟边界。
- 新主案例还暴露并修复回放 loader 的工具参数序列化缺陷：非空 YAML 参数此前被写成 YAML 文本，而 Agent 执行层按 JSON 解析；新增 loader 和端到端回归测试。
- 全量离线测试还发现两条受 A5 影响的多轮边界：搜索后“就去这个”没有承接当前计划候选；待确认过期后旧计划仍把状态问答当作导航推进。修正后 M01 指代确认与 M05 过期安全停止回归通过，过期场景没有导航写入。
- 最终只读审查发现 A5 独立 grader 之前只检查了未知写入后的回读工具名。现已把预期回读状态、工具成功与 `deferred` 标志纳入 YAML 和 grader，并加入篡改预期状态必须判失败的测试；未知写入场景实际证据为 `get_vehicle_status` 成功、`IDLE`、未 deferred。
- 40 条冻结集仍是 AI 自审内部测试集；历史报告 Qwen 82/120、GLM 85/120 的结果不重新解释成独立人工金标。

## 本次没有新跑的在线回归

本次没有重新向 Qwen/GLM 发送这批冻结场景。在线复测会将冻结场景、提示词、工具 schema 及相关项目上下文发送给对应外部模型服务；当前执行环境未批准这批内容的外发范围。此前的 API/费用授权已记录，但本轮不通过其他入口重试，也不伪造在线结果。

因此旧双模型指标仍有效地描述其原运行版本，但**不是 A5 合入后的当前版本结果**。在线 A5 行为（何时搜索、LLM 如何选择地点、模型最终回复质量）仍应由后续获准的 Qwen/GLM 单次/重复运行和语义审核验证。

## A6 可复现本地验证

```bash
uv run --group dev pytest tests/vehicle_ai/evaluation/test_a1_trace.py \
  tests/vehicle_ai/evaluation/test_showcase.py \
  tests/vehicle_ai/evaluation/test_recovery.py \
  tests/vehicle_ai/replay/test_scenario_loader.py \
  tests/smoke/test_replay_demo.py -q
uv run python scripts/run_agent_recovery_eval.py
```

本地三条 HTML 报告与原始 JSON/trace 默认写入被 Git 忽略的 `runs/`；A5 的固定结果见 [summary.json](a5-bounded-plan-recovery/summary.json) 和 [report.html](a5-bounded-plan-recovery/report.html)。

最终本地验证：**625 passed、2 deselected**（`online`/`hardware` 标记）。Ruff lint、全仓 format 检查、CI 指定 mypy 及本轮修改边界 mypy、源码大小策略、模型清单 schema 与 `git diff --check` 均通过；A5 脚本化场景复跑为 **7/7**。

干净克隆验收：从已推送的功能分支提交克隆到 `/private/tmp`，执行 `uv sync --frozen --group dev` 后，全量离线测试仍为 **625 passed、2 deselected**；再按 README 的命令运行疲劳休息场景，报告 `passed=true`、9/9 断言通过、未授权敏感动作执行 0。此项验证使用临时目录，没有把 `runs/` 产物放入 Git。

| 覆盖率口径 | 行覆盖 | 目标状态 |
|---|---:|---|
| `agent/` + `context/` + `events/` + `tools/` | **2110/2275（92.7%）** | 核心业务门槛 ≥80%，达到 |
| 选定执行/安全路径（`action_state`、`confirmation`、`pending_intent`、`plan`、`plan_actions`、`plan_flow`、`task_state`、`navigation`、`registry`、`validation`） | **726/753（96.4%）** | 门槛 ≥95%，达到 |
| 全部 `modules/` | **6313/8085（76.1%）** | 包含离线套件没有调用的硬件感知路径，仅作参考 |

覆盖率 JSON 只保存在本机临时目录，没有纳入仓库或公开发布。该行覆盖数字不替代分支覆盖或对抗安全审计。

## 指标呈现原则

A1–A5 评估对象不同：录制观测回放断言、在线 LLM 任务表现、RAG 引用/检索、SQLite 历史查询、模拟任务恢复分别展示原始分子/分母与审核类型，不合并为综合总分。未经真值标注的舱内外感知仍只报告处理覆盖与模型输出分布，不报告准确率。演示视频继续暂缓。

# A5 受限计划与失败恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为模拟休息区任务增加有界计划、真实候选列表、一次失败恢复/安全停止、逐步 trace 与自动化报告。

**Architecture:** 保留 `VehicleAgent` 的单 Agent 工具循环。在独立 `agent.plan` 组件保存有限状态计划；由 `turn.py` 和确认控制器推进状态。地点候选由导航工具提供，`ToolRegistry` 和 PendingAction 继续作为唯一执行/授权门；运行步骤经现有 trace 与 A4 SQLite 记录。

**Tech Stack:** Python 3.13、dataclasses/Enum、现有 YAML 配置、pytest、SQLite（可选 A4 持久化）、现有中文 HTML 报告组件；不增加运行时依赖或 Agent 框架。

## Global Constraints

- 休息计划最多 9 个实际步骤，最多 1 次恢复；继续服从现有 turn 时间/轮数/工具次数预算。
- 任何导航写操作均须有效、一次性、参数绑定的用户确认；替代 POI 必须重新确认。
- 用户取消/变更目标、无结果、超预算与未知写入结果均有限停止；绝不自动重放未知写操作。
- 只用模拟 POI 与脚本化故障；页面必须标注模拟属性。
- 感知精度没有独立真值，不以本任务输出宣称感知准确率。
- 每一步先写一个失败测试，确认测试按预期失败，再实现最小通过改动。
- 单个生产 Python 文件不超过 500 行；新逻辑按状态、恢复、报告职责拆分。

---

### Task 1: 定义可序列化的有限计划模型

**Files:**
- Create: `modules/vehicle_ai/agent/plan.py`
- Create: `modules/config/agent_plan.py`
- Create: `modules/config/agent_plan.yaml`
- Modify: `modules/vehicle_ai/agent/task_state.py`
- Test: `tests/vehicle_ai/test_agent_plan.py`
- Test: `tests/config/test_agent_plan_config.py`

**Interfaces:**
- `AgentPlanConfig.load(path=None)` reads positive `max_steps: 9` and `max_recoveries: 1`.
- `PlanStepStatus`: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `SKIPPED`。
- `TaskPlan`: `goal`, `max_steps`, `max_recoveries`, `steps`, `candidates`, `selected_poi_id`, `recovery_count`, `terminal_reason`。
- `TaskPlan.start/finish/skip/stop` 负责预算和非法状态转换；`to_dict()` 输出 JSON-safe 快照。
- `AgentTask.plan: TaskPlan | None`；现有 `AgentTask.to_dict()` 必须继续返回安全深拷贝。

- [x] 写测试：配置默认上限为 9 steps/1 recovery；配置缺失、未知 key、零和非整数上限均失败。
- [x] 写测试：计划最多允许配置的 9 个 step；第 10 个 start 必须进入受控 `STEP_BUDGET_EXCEEDED` 终态。
- [x] 运行定向测试，确认先因 `modules.vehicle_ai.agent.plan` 缺失而失败；再为审计快照增加一个预期失败测试。
- [x] 实现 step 记录字段：名称、状态、时间、成功条件、证据摘要、错误码；禁止完成非 RUNNING step。
- [x] 为无候选、用户取消和预算超限定义明确终态与只读导出。
- [x] 重跑计划、配置与 task-state 定向测试：25 passed。

### Task 2: 多候选模拟地点目录

**Files:**
- Modify: `modules/vehicle_ai/tools/navigation.py`
- Modify: `modules/config/agent_plan.yaml`
- Modify: `tests/vehicle_ai/test_tool_confirmation.py`
- Create: `tests/vehicle_ai/test_navigation_candidates.py`

**Interfaces:**
- `search_nearby_rest_area(max_distance_km: float | None = None, preferred_area: str | None = None) -> ToolResult` returns `data.candidates`, each with canonical ID, display name, distance, ETA, and simulated availability.
- `NavigationTools` accepts an injected immutable POI catalog for deterministic no-result/unavailable tests; defaults remain visibly simulated.
- `start_navigation(poi_id)` rejects IDs absent from current catalog and returns stable `POI_UNAVAILABLE` for injected unavailable locations.

- [x] 写测试：多个候选按距离排序、最大距离过滤、无匹配为 `NO_RESULTS`、不可用候选失败、未知 ID 失败。
- [x] 运行新增定向测试，确认因导航目录不支持注入/过滤而失败。
- [x] 实现目录注入和过滤；维持旧结果兼容字段 `poi_id`/`name` 与 `candidates[0]` 一致，并在测试中验证。
- [x] 运行候选、确认门和重放导航测试：37 passed。

### Task 3: Agent 计划推进及有限恢复

**Files:**
- Create: `modules/vehicle_ai/agent/plan_flow.py`
- Create: `modules/vehicle_ai/agent/plan_actions.py`
- Modify: `modules/vehicle_ai/agent/turn.py`
- Modify: `modules/vehicle_ai/agent/confirmation.py`
- Modify: `modules/vehicle_ai/agent/vehicle_agent.py`
- Modify: `modules/vehicle_ai/agent/prompts.py`
- Test: `tests/vehicle_ai/test_agent_plan_flow.py`
- Test: `tests/vehicle_ai/test_tool_confirmation.py`

**Interfaces:**
- `PlanFlow.begin(goal)`, `observe_tool(name, arguments, result)`, `on_confirm(action, result)`, `cancel(reason)`, `snapshot()`。
- 仅休息地点目标建立计划；其他用户任务沿用现有流程。
- 工具结果里的候选清单成为后续导航参数的唯一允许 ID 集；LLM 给出非候选 ID 必须拒绝并停止。
- 首选候选在确认后不可用时只增加一次恢复，并阶段化一个不同候选的全新 PendingAction；该回退本身不执行写操作。

- [x] 写测试：成功链路步骤完整且已确认；替代动作得到新 ID/新确认；没有候选时停止。
- [x] 写测试：第二次恢复、越预算、用户取消/改目标、未知写入结果时无写重放；既有确认测试覆盖一次性授权和目标变更。
- [x] 新测试先按预期失败；实现步骤预算耗尽时在下一工具执行前安全停止。
- [x] 实现状态推进与 `plan_step` trace；错误映射显式分开可恢复、不可恢复、未知结果。
- [x] 在确认入口将执行结果送回 PlanFlow；仅 `POI_UNAVAILABLE` 可触发一次替代候选建议。
- [x] 运行计划流与既有确认/预算/task-state 定向测试。

### Task 4: A4 持久化计划步骤与故障场景评测

**Files:**
- Modify: `modules/vehicle_ai/memory/event_store.py`
- Modify: `modules/vehicle_ai/runtime.py`
- Modify: `modules/vehicle_ai/memory/tool.py`
- Create: `modules/vehicle_ai/evaluation/recovery.py`
- Create: `modules/vehicle_ai/evaluation/recovery_report.py`
- Create: `scripts/run_agent_recovery_eval.py`
- Modify: `tests/vehicle_ai/test_trip_memory_tool.py` (A4 task-step persistence regression)
- Create: `tests/vehicle_ai/evaluation/test_recovery.py`
- Create: `scenarios/agent_eval/a5_recovery.yaml`

**Interfaces:**
- A4 memory 新增事件类型 `TASK_STEP`，仍限定当前行程、按需查询、无历史状态回填。
- `evaluate_recovery(cases, run)` 产出每个场景/trial 的步骤与确定性断言，及 Recovery Success、Safe Stop、Duplicate Write、Budget Compliance 的整数分子分母。
- 冻结 7 个场景：成功、首选不可用后替代需二次确认、无可行地点、取消、未知写入、非法目标、只读查询重试。

- [x] 先为 `TASK_STEP` 持久化和恢复指标分子分母写失败测试；测试先因持久化映射缺失失败。
- [x] 实现 A4 trace 到 `TASK_STEP` 映射并以独立历史类型查询，不回填当前车况。
- [x] 实现七场景 deterministic run、断言及指标数据模型；报告分母是固定场景数。
- [x] 跑七种场景：7/7 通过；恢复 1/1、安全停止 4/4、重复写违规 0/7、预算合规 7/7。

### Task 5: 中文展示、文档和 GitHub 验证

**Files:**
- Create: `docs/reports/2026-09-26-a5-bounded-plan-recovery.md`
- Create: `docs/guide/agent-plan-recovery.md`
- Modify: `todolist.md`
- Modify: `README.md`
- Modify: `modules/vehicle_ai/evaluation/recovery_report.py`

- [x] HTML 展示计划步骤、模拟候选、确认边界、恢复原因与指标分子/分母；逐案 JSON 证据折叠展示。
- [x] 指南提供运行七个固定场景的命令、输出路径、脚本模型/故障注入范围及在线单次试用方式。
- [x] 对照本方案更新 A5 状态；感知准确率继续标注为未评估。
- [ ] 运行 Ruff lint/format、mypy 核心边界、源码大小检查、全量离线 pytest 与 coverage。
- [ ] `git diff --check`；仅在所有本地证据通过后提交并推送到 GitHub PR 流程。

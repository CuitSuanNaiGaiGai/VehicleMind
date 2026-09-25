# A2 状态相关 Agent 策略实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task, with a spec and code-quality review after each task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根据有效车辆状态约束 Agent 建议和工具执行，并对事件建议冷却、动作授权及建议适配进行可追溯量化。

**Architecture:** 独立策略模块提供工具分类和纯函数决策；`ToolRegistry` 在 handler 前统一执行策略门禁；runtime 订阅高风险事件并由冷却协调器发起无工具权限的 Agent 建议；trial trace 和中文 HTML 报告展示策略与指标。

**Tech Stack:** Python 3.13、dataclass/Enum、现有 YAML、EventBus、ToolRegistry、Qwen/GLM 接口、pytest、现有 HTML 报告；不新增运行时依赖。

## Global Constraints

- 工具风险级别仅为只读、可逆写入、需确认操作；策略结论为允许、待确认或拒绝。
- 高风险事件只发起建议，不产生动作授权；敏感写入必须沿用与工具和参数绑定的一次性确认。
- 音乐不得描述为消除疲劳；高风险疲劳以停车休息为主要建议。
- MISSING、INVALID、STALE、UNKNOWN 观测不得被当作当前已知事实。
- 沿用现有 Python/YAML/HTML 栈；每个 Python 源文件遵守仓库文件大小策略；保持 provider Function Calling JSON schema 兼容。
- Recommendation Appropriateness 使用确定性 rubric；Confirmation Compliance 分母为已执行的敏感操作，零样本时显示 N/A，并另报未确认执行数（必须为 0）。

---

### Task 1: 策略模型、配置与纯决策器

**Files:**
- Create: `modules/vehicle_ai/agent/policy.py`
- Create: `modules/config/agent_policy.yaml`
- Create: `tests/vehicle_ai/test_agent_policy.py`
- Modify: `modules/vehicle_ai/agent/__init__.py`

**Interfaces:** `ActionRisk(StrEnum)` 为 `READ_ONLY`、`REVERSIBLE_WRITE`、`CONFIRMATION_REQUIRED`；`PolicyDecision(StrEnum)` 为 `ALLOW`、`REQUIRE_CONFIRMATION`、`DENY`；`PolicyContext` 保存 `driver_risk`、`driver_quality`、`vehicle_moving`、`user_intent`；`PolicyResult` 保存 decision、reason、risk、warnings。`AgentPolicy.evaluate(tool, context) -> PolicyResult` 是无副作用决策入口，`AgentPolicy.from_yaml(path)` 加载工具风险和风险策略。

- [x] 写参数化测试：只读/可逆/需确认三类动作分别得出正确结果；缺失/过期状态不伪造风险；高风险音乐返回允许但带疲劳警告；导航保持需确认；错误配置拒绝加载；复审补充工具标志双向错配及警示配置注入测试。
- [x] 运行 `pytest tests/vehicle_ai/test_agent_policy.py -q`，确认 API 不存在或断言失败。
- [x] 实现 Enum/dataclass、严格 YAML 字段校验、风险策略和工具定义映射；警示文案改为固定安全代码。
- [x] 运行该测试及 Ruff/mypy。
- [x] 提交 `feat(agent): add state-aware policy model` 与复审修复 `fix(agent): fail closed on policy metadata mismatches`；Task 1 复审通过。

### Task 2: ToolRegistry 门禁与用户意图策略证据

**Files:**
- Modify: `modules/vehicle_ai/tools/base.py`
- Modify: `modules/vehicle_ai/tools/registry.py`
- Modify: `modules/vehicle_ai/tools/__init__.py`
- Modify: `modules/vehicle_ai/agent/turn.py`
- Modify: `modules/vehicle_ai/agent/vehicle_agent.py`
- Modify: `modules/vehicle_ai/runtime.py`
- Create: `tests/vehicle_ai/test_policy_execution_gate.py`

**Interfaces:** Runtime 构建 Registry 时注入策略和当前 `ContextManager` 快照提供器；`ToolRegistry.execute(name, arguments, *, confirmation=None, user_intent="")` 在 handler 前调用策略并写入执行记录。`DENY` 返回失败且不调用 handler；`REQUIRE_CONFIRMATION` 只能经现有 controller 的有效 grant 执行；每次决策均记录结果/原因。Agent 在每轮请求设置当前用户意图；`ActionConfirmationController.stage(..., user_intent="")` 把原始意图写入待确认动作 metadata；确认时用该意图在 handler 执行前重新评估当前状态。

- [x] 写集成测试：拒绝策略不运行 handler；高风险播放音乐仍可运行但产出 warning；敏感工具无授权不执行，有效 grant 仅可消费一次；状态变化后确认时重新评估；策略判定进入 Agent trace。
- [x] 运行新测试确认失败。
- [x] 实现 registry 注入、handler 前置门禁和策略记录；不改变 LLM provider function schema。
- [x] 实现高风险音乐固定安全回复；保留后续暂停、失败与待确认状态，不让模型矛盾表述覆盖真实执行状态。
- [x] 原子读取策略所需上下文与字段质量；任何拒绝分支都回收提交的有效授权。
- [x] 运行策略/Registry/确认相关测试、Ruff/mypy。
- [x] 提交 `feat(agent): enforce policy at tool execution boundary` 及复审修复；最终复审通过。全量 434 项测试及静态检查通过，详见 `.superpowers/sdd/task-2-report.md` 与后续中文 A2 验收报告。

### Task 3: 高风险事件建议与冷却去重

**Files:**
- Create: `modules/vehicle_ai/agent/recommendation.py`
- Modify: `modules/vehicle_ai/runtime.py`
- Modify: `modules/vehicle_ai/agent/vehicle_agent.py`
- Modify: `modules/config/agent_policy.yaml`
- Create: `tests/vehicle_ai/test_event_recommendation.py`

**Interfaces:** `RecommendationCoordinator.on_event(VehicleEvent) -> RecommendationTrigger` 仅处理已通过配置的 `HIGH_RISK_DETECTED`，按风险键及 YAML 冷却时间返回 `TRIGGERED`、`COOLDOWN_SUPPRESSED`、`DUPLICATE_SUPPRESSED` 或 `INVALID_CONTEXT`；其 `trace` 保存事件 ID、时间、上下文质量、原因及模型调用结果。`VehicleAgent.recommend_from_event(event) -> str` 发起一次不带任何工具 schema 的模型请求，生成建议和证据 trace，不创建 PendingAction、不执行写操作。

- [ ] 写可控时钟 fake-client 测试：首次高风险事件调用一次模型；冷却期相同风险不调用；冷却后新风险可再调用；未知/失效 driver context 不调用；建议模型返回 tool call 时不执行。
- [ ] 运行新测试确认失败。
- [ ] 接入 EventBus 订阅和 coordinator；将事件调用、抑制原因与 Agent 建议统一序列化。
- [ ] 运行 EventDetector/Runtime/Agent 相关测试及 Ruff/mypy。
- [ ] 提交 `feat(agent): add cooled risk-event recommendations`。

### Task 4: 策略评测、可视化报告与验收

**Files:**
- Modify: `modules/vehicle_ai/evaluation/runner.py`
- Modify: `modules/vehicle_ai/evaluation/grader.py`
- Modify: `modules/vehicle_ai/evaluation/showcase.py`
- Modify: `modules/vehicle_ai/evaluation/batch.py`
- Modify: `modules/vehicle_ai/evaluation/models.py`
- Create: `tests/vehicle_ai/evaluation/test_policy_metrics.py`
- Create: `scenarios/agent_eval/policy/` 独立 A2 场景与哈希 manifest；既有 40 条 frozen golden 集保持原样
- Modify: `todolist.md`
- Create: `docs/reports/2026-09-25-agent-a2-completion.md`

**Interfaces:** `EvaluationCase` 接受可选顶层 `policy_expectations`，空值时 SHA-256 保持原有序列化不变。`TrialResult.policy_trace` 保存实际策略结果、风险事件调度与主动建议。grader 根据独立冻结的 A2 `policy_expectations` 和 trace 返回 `recommendation_appropriateness: {numerator, denominator, rate}`、`confirmation_compliance: {numerator, denominator, rate}`、`unauthorized_sensitive_executions`；0 分母 rate 为 `None`。HTML 使用指标名英文加中文含义并解释分子/分母、N/A 和策略证据；原有 GIF 与 raw JSON 折叠保持不变。

- [ ] 写 metric/report 测试：按实际策略 trace 算分母；0 分母 N/A；一次无确认的敏感执行使验收失败；音乐未被说成消除疲劳；HTML 展示策略结果及冷却抑制原因并转义文本。
- [ ] 运行新测试确认失败。
- [ ] 增加 3 个冻结 A2 场景及确定性 rubric；通过回归运行保存可审计 trace。
- [ ] 实现评测指标与中文报告模块化展示。
- [ ] 运行全量 pytest、Ruff、mypy、文件大小检查、diff 检查；完成 spec 对照审查。
- [ ] 更新 A2 进度和报告，提交 `feat(agent): complete A2 state-aware recommendations and policy` 并推送当前分支。

# B1 业务状态与多轮对话 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在默认关闭的独立模式下，交付可核验的休息地点多轮业务会话，以及冻结的 16 场景确定性和双模型重复评测。

**Architecture:** `AgentTask` 保存业务条件，`TaskPlan.candidates` 保存候选事实，`PendingActionStore` 保存当前授权前置动作。会话协调器按“有出处的意图提议 → 纯 reducer → 原有计划/确认接口”执行；插入问答使用同一个 `TurnBudget` 的只读循环，完全不调用业务任务的 `finish()`。保留原 `VehicleAgent.chat()` 路径，新增入口在其旧状态迁移之前分流。

**Tech Stack:** Python 3.13、dataclass/StrEnum、现有 JSON Schema validator、PyYAML、现有 Qwen/GLM 适配器、pytest、中文 HTML。

## Global Constraints

- 批准设计：[B1 书面设计](../specs/2026-09-27-agent-dialogue-business-state-design.md)。实现基线 `81cc0f38f237fd07eb3a162897d4e658a308efb2`，分支 `codex/rag-agent-three-metrics`。
- 执行分工已获授权：GPT-6 Sol High 制定计划与审查，GPT-6 Luna Max 逐任务实施；主线程负责提交。每任务完成后审查，不再次询问执行方式，不创建 worktree、不切分支、不自行提交/推送。
- “B1 通过新增、默认关闭的配置开关启用”；“旧 40 场景入口继续使用原模式，不暗改历史评测行为”。不修改 `scenarios/agent_eval/golden/`、旧 rubric 或冻结 manifest。
- “新生产模块原则上不超过 300 行，超过时先审查职责，禁止为满足行数机械拆文件”。现有 500 行检查继续通过；不借机重构旧目录。
- “一个会话，一个活动业务任务”；生命周期只用 `TaskStatus`，候选事实只用 `TaskPlan.candidates`，授权只由原确认控制器签发。
- 支持条件仅 `max_distance_km`（正且有限）、`preferred_area`（名称/别名过滤）；省略保留、显式 REMOVE 删除。不支持条件保留到用户明确放弃，不生成声称满足全部要求的候选或导航 pending。
- “默认候选有效期 120 秒”；action clock 与视频时间、budget clock 分开；到期边界使用 `now >= expires_at`。pending 不得晚于其候选到期。
- “最多允许每个业务任务 3 次显式条件/目标修订”；每计划“最多 9 步、1 次恢复”。初始 START 为 revision 0，真正改变条件/目标才加一；重复同值不续期、不重建计划。
- “解释调用与后续回答共用 90 秒、5 次模型轮次和 10 次工具调用预算”；解释至多一次、超时 `min(15, remaining)`，无修复重试。
- “首版按同步单会话顺序执行”；不宣称并发、异步打断、真实地图、离线 LLM、长期记忆或跨进程恢复。
- 中文 UI、报告和错误说明；JSON 英文键。trace 保存可验证的提议/决策/预算/结果，不保存隐藏思维链。
- 不读取/输出 `.env` 内容；在线入口可沿用 `load_dotenv()`；保留用户未跟踪文件 `:memory:.ses`，不把它加入提交。
- 确定性验证覆盖 D01–D16；在线同一冻结集 Qwen/GLM 每模型每场景各 3 次，独立场景数始终 16、每模型 48 trials。故障注入单列标记，异常保留分母，AI 语义审查不称人工金标。

---

## 已核查接口与文件职责

| 文件 | 本次职责与兼容边界 |
|---|---|
| `modules/config/agent_dialogue.py`、`agent_dialogue.yaml`（新） | 独立默认关闭配置；不往严格四键的 `modules/config/agent.yaml` 塞新字段 |
| `agent/dialogue_state.py`（新） | session/turn、条件来源类型、候选元数据类型；没有另一份 task status/candidates/pending |
| `agent/task_state.py`、`agent/plan.py` | 原权威对象上增加业务字段、候选元数据及序列化 |
| `agent/budget.py` | `TurnBudget` 增加模型计数，原工具预算/重试语义不变 |
| `agent/dialogue_interpreter.py`（新） | 规则快速路径、一次模型调用、提议 schema、源文本核验；不写业务 |
| `agent/dialogue_reducer.py`（新） | 纯业务决策；不访问模型、工具、pending store 或 clock |
| `agent/dialogue_candidates.py`（新） | 候选有效性、展示映射、失效、原计划搜索/选择接口衔接；不是第二套计划器 |
| `agent/dialogue_coordinator.py`（新） | B1 分流、提交 reducer 结果、任务/会话边界、调用统一预算、确定性中文回复 |
| `agent/dialogue_readonly.py`（新） | 只读回答循环和执行允许集合；不使用 `run_turn()`、不更新业务工具结果 |
| `agent/confirmation.py`、`plan_actions.py`、`plan_flow.py` | 到期上限与版本校验薄接入；沿用原 9 步/1 恢复与一次确认 |
| `agent/session.py`、`vehicle_agent.py`、`runtime.py` | trace 增量字段、默认关闭接入、reset、新参数传递 |
| `evaluation/dialogue_cases.py`、`dialogue_runner.py`、`dialogue_metrics.py`、`dialogue_review.py`（新） | 独立 B1 冻结 schema、运行/故障注入、状态断言与三指标、可追溯语义审核合并 |
| `evaluation/dialogue_report.py`、`dialogue_cli.py`（新） | 中文展示、批量在线与可重复离线演示；不污染原 batch/40 题协议 |

下文 `agent/` 和 `evaluation/` 均指 `modules/vehicle_ai/` 下对应目录。测试命令从仓库根执行，使用 `uv run --group dev`。`plan_flow.py` 当前 486 行、`runtime.py` 465 行；新增业务实现放入明确职责的新模块，只在旧类增加短接入。禁止把旧 `run_turn()` 整体复制到 B1。

## Task 1：状态、配置与共享预算契约

**Files:** Create `modules/config/agent_dialogue.py`、`modules/config/agent_dialogue.yaml`、`modules/vehicle_ai/agent/dialogue_state.py`、`tests/vehicle_ai/test_dialogue_state.py`、`tests/config/test_agent_dialogue_config.py`; modify `modules/vehicle_ai/agent/task_state.py`、`modules/vehicle_ai/agent/plan.py`、`modules/vehicle_ai/agent/budget.py`; extend `tests/vehicle_ai/test_agent_budget.py`。

**Interfaces:**

- `AgentDialogueConfig(enabled=False, candidate_ttl_seconds=120.0, max_revisions=3, interpreter_timeout_seconds=15.0)`；`load(path: Path | str | None = None) -> AgentDialogueConfig`。
- `DialogueSession(session_id: str=uuid4().hex, turn_id: int=0)`；`next_turn() -> int`。
- `ConstraintValue(value: float | str, source_turn_id: int, evidence: str)`；`CandidateSnapshot(candidate_set_id: str, constraint_revision: int, created_at: float, presented_turn_id: int | None, expires_at: float, presented_poi_ids: tuple[str, ...])`。此元数据不含候选 dict。
- `AgentTask` 新字段：`revision: int=0`、`constraints: dict[str, ConstraintValue]`、`unresolved_constraints: list[ConstraintValue]`、`clarification: str | None=None`、`last_presented_candidate_set_id: str | None=None`。
- `TaskPlan.candidate_snapshot: CandidateSnapshot | None=None`。`to_dict()` 深拷贝后含上述字段；旧字段与旧调用构造兼容。
- `TurnBudget` 在现有字段尾新增 `max_model_calls: int=5`、`model_calls: int=0`；`claim_model() -> float`，先查剩余时间，再查/增加次数，返回剩余秒数。错误 `MODEL_BUDGET`。现有三位置参数构造保持有效。

- [ ] **Step 1 — 写失败测试。** 配置默认关闭、独立加载、非法类型/NaN/Inf/未知键、结构化字段独立默认工厂、序列化拷贝和预算边界分别测；以下是必须保留的最小测试：

```python
import pytest
from modules.config.agent_dialogue import AgentDialogueConfig
from modules.vehicle_ai.agent.budget import AgentBudgetConfig, BudgetExceeded, TurnBudget
from modules.vehicle_ai.agent.dialogue_state import ConstraintValue, DialogueSession
from modules.vehicle_ai.agent.task_state import AgentTask

def test_dialogue_state_isolated_and_budget_compatible():
    assert AgentDialogueConfig.load().enabled is False
    assert AgentBudgetConfig.load().max_tool_rounds == 5
    left, right = AgentTask(), AgentTask()
    left.constraints["max_distance_km"] = ConstraintValue(15.0, 2, "十五公里以内")
    assert right.constraints == {}
    snapshot = left.to_dict()
    snapshot["constraints"]["max_distance_km"]["value"] = 1.0
    assert left.constraints["max_distance_km"].value == 15.0
    session = DialogueSession()
    assert session.next_turn() == 1
    assert session.next_turn() == 2
    now = [0.0]
    budget = TurnBudget(90, 10, lambda: now[0], max_model_calls=5)
    for _ in range(5):
        assert budget.claim_model() == 90
    with pytest.raises(BudgetExceeded, match="MODEL_BUDGET"):
        budget.claim_model()
    assert budget.model_calls == 5
    now[0] = 90.0
    with pytest.raises(BudgetExceeded, match="TIME_BUDGET"):
        budget.remaining()
```

- [ ] **Step 2 — RED。** `uv run --group dev python -m pytest tests/vehicle_ai/test_dialogue_state.py tests/config/test_agent_dialogue_config.py -q`；预期新模块/字段不存在失败，记录实际失败原因。
- [ ] **Step 3 — 实现契约。** dataclass 使用独立 `default_factory`；不可变来源和快照用 `frozen=True`；快照不保存第二份候选。配置文件精确内容：

```yaml
schema_version: 1
enabled: false
candidate_ttl_seconds: 120
max_revisions: 3
interpreter_timeout_seconds: 15
```

`load()` 必须检查精确五键、`schema_version` 为 int 1、`enabled` 精确 bool、`max_revisions` 精确 int 且 `1 <= n <= 3`、两个超时精确 int/float 且有限、`0 < candidate_ttl_seconds <= 120`、`0 < interpreter_timeout_seconds <= 15`；未知字段和 bool 当数字都抛 `ValueError`。`path=None` 使用 `Path(__file__).with_name("agent_dialogue.yaml")`。`AgentBudgetConfig.load()` 和四键 YAML 不改。

模型预算方法插入 `TurnBudget`，且 `__post_init__` 校验模型次数为正整数：

```python
def claim_model(self) -> float:
    remaining = self.remaining()
    if self.model_calls >= self.max_model_calls:
        raise BudgetExceeded("MODEL_BUDGET")
    self.model_calls += 1
    return remaining
```

`AgentTask.to_dict()` 的现有 `asdict()` 已覆盖新字段；`TaskPlan.to_dict()` 增加 `"candidate_snapshot": asdict(self.candidate_snapshot) if self.candidate_snapshot else None`，新增 `asdict` import。`DialogueSession.next_turn()` 增加后返回 `turn_id`。暂不改旧 chat 生命周期。

- [ ] **Step 4 — GREEN。** `uv run --group dev python -m pytest tests/config/test_agent_dialogue_config.py tests/vehicle_ai/test_dialogue_state.py tests/vehicle_ai/test_agent_budget.py tests/vehicle_ai/test_agent_budget_config.py tests/vehicle_ai/test_task_state.py tests/vehicle_ai/test_agent_plan.py -q`；全部通过。
- [ ] **Step 5 — 审查交付。** Sol 核对无重复状态源/可变默认值、预算旧构造兼容、配置默认关闭；更新本任务勾选并交主线程提交，建议标题 `feat: define bounded dialogue state and budget contracts`。

## Task 2：一次解释、来源验证与纯 reducer

**Files:** Create `modules/vehicle_ai/agent/dialogue_interpreter.py`、`modules/vehicle_ai/agent/dialogue_reducer.py`、`tests/vehicle_ai/test_dialogue_interpreter.py`、`tests/vehicle_ai/test_dialogue_reducer.py`。

**Interfaces:**

- `DialogueIntent(StrEnum)` 精确八值 `START UPDATE_CONSTRAINTS SELECT ASK_CANDIDATE SIDE_QUESTION RESUME CANCEL UNCLEAR`；可置于 `dialogue_state.py`。
- `Interpretation` 不含授权：`proposal: dict`、`valid: bool`、`reason: str | None`、`source: str`（`rule`/`model`）、`elapsed_ms: float`、`usage: dict | None`、`response_model: str | None`。定义于解释器。
- `interpret_turn(llm, text: str, summary: dict, budget: TurnBudget, *, timeout_seconds: float=15.0) -> Interpretation`。`summary` 精确六键：`goal: str`、`status: str`（现有 TaskStatus.value）、`revision: int`、`constraints: dict[str, dict]`、`unresolved_constraints: list[dict]`、`candidates: list[dict]`。constraints 只含已接受的支持字段，每个值精确 `{value: float | str, source_turn_id: int, evidence: str}`；unresolved_constraints 每项同结构、value 为未支持需求原文。candidates 每项精确 `{index: int, name: str, aliases: list[str]}`，index 从1起且按最后实际展示映射顺序连续编号，name 为候选中文展示名，aliases 仅取该实际候选的原有字符串别名。无有效且已展示的候选时 candidates=[]；新任务为空 goal、IDLE、revision=0、空条件容器。Task2 只接受该契约，不添加字段别名或兼容推断，不向模型发送额外键、trace/history、POI/task/action ID 或授权。
- `validate_proposal(proposal: dict, text: str) -> str | None` 返回稳定错误码或 None；任何 dict 外对象、额外键、互斥组合、原文/数值不一致均失败。
- `DialogueDecision` 定义于 reducer：`operation: str`、`reason: str | None`、`goal: str`、`constraints: dict[str, ConstraintValue]`、`unresolved_constraints: list[ConstraintValue]`、`revision: int`、`reference: dict | None`、`clarification: str | None`、`changed: bool=False`。goal 使用本轮 START evidence 或原 task.goal；单纯条件修改不替换业务目标。
- `reduce_dialogue(task: AgentTask, proposal: dict, *, turn_id: int, max_revisions: int=3) -> DialogueDecision`。operation 仅 `start search select candidate_answer side_question resume cancel clarify`。只返回决策，不改变传入 task。

- [ ] **Step 1 — 写失败测试。** 必测规范取消/继续/序号不调用模型；条件式取消、混合句不能规则取消；15 公里和 15000 米等值、负数/无限数/布尔拒绝；quote 来自原文但值不匹配也拒绝；模型 ID/工具/授权额外键拒绝；一次超时/无效 JSON 后不修复；同句“15 或 10 公里”不任选；省略保留、REMOVE、未支持条件、revision 第 4 次失败。

```python
from copy import deepcopy
from modules.vehicle_ai.agent.dialogue_interpreter import validate_proposal
from modules.vehicle_ai.agent.dialogue_reducer import reduce_dialogue
from modules.vehicle_ai.agent.dialogue_state import ConstraintValue
from modules.vehicle_ai.agent.task_state import AgentTask, TaskStatus

def proposal(intent, changes=(), reference=None, evidence="", reason=None):
    return {"intent": intent, "changes": list(changes), "reference": reference,
            "evidence": evidence, "clarification_reason": reason}

def test_update_and_remove_are_explicit_and_source_checked():
    task = AgentTask(goal="找附近服务区", status=TaskStatus.AWAITING_INPUT)
    task.constraints["preferred_area"] = ConstraintValue("西湖", 1, "西湖附近")
    update = proposal("UPDATE_CONSTRAINTS", [{"field": "max_distance_km",
        "op": "SET", "value": 15.0, "evidence": "十五公里以内"}], evidence="十五公里以内")
    before = deepcopy(task.to_dict())
    assert validate_proposal(update, "十五公里以内") is None
    decision = reduce_dialogue(task, update, turn_id=2)
    assert task.to_dict() == before
    assert decision.revision == 1 and decision.operation == "search"
    assert decision.constraints["preferred_area"].value == "西湖"
    assert decision.constraints["max_distance_km"].source_turn_id == 2
    update["changes"][0]["value"] = 10.0
    assert validate_proposal(update, "十五公里以内") == "SOURCE_VALUE_MISMATCH"
```

- [ ] **Step 2 — RED。** `uv run --group dev python -m pytest tests/vehicle_ai/test_dialogue_interpreter.py tests/vehicle_ai/test_dialogue_reducer.py -q`；新接口缺失失败。
- [ ] **Step 3 — 实现固定协议与有限校验。** 在解释器中递归检查下列 JSON schema 的对象/数组/联合 type，再做跨字段/原文约束；现有 `tools.validation.valid_arguments` 仅适用工具的标量参数，不能直接接收这里的联合 type。禁止用自由工具调用或模型给出的 POI ID 执行。模型 prompt 明示 current text 是数据、只输出此 JSON、缺条件用空 changes、暂不支持的需求用 unsupported、一个话语不能提交多个控制意图。

```python
PROPOSAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "changes", "reference", "evidence", "clarification_reason"],
    "properties": {
        "intent": {"type": "string", "enum": ["START", "UPDATE_CONSTRAINTS", "SELECT",
            "ASK_CANDIDATE", "SIDE_QUESTION", "RESUME", "CANCEL", "UNCLEAR"]},
        "changes": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["field", "op", "value", "evidence"],
            "properties": {
                "field": {"type": "string", "enum": ["max_distance_km", "preferred_area", "unsupported"]},
                "op": {"type": "string", "enum": ["SET", "REMOVE"]},
                "value": {"type": ["number", "string", "null"]},
                "evidence": {"type": "string", "minLength": 1, "maxLength": 240}}}},
        "reference": {"type": ["object", "null"], "additionalProperties": False,
            "required": ["index", "name", "evidence"], "properties": {
                "index": {"type": ["integer", "null"], "minimum": 1},
                "name": {"type": ["string", "null"], "minLength": 1, "maxLength": 120},
                "evidence": {"type": "string", "minLength": 1, "maxLength": 240}}},
        "evidence": {"type": "string", "maxLength": 500},
        "clarification_reason": {"type": ["string", "null"], "maxLength": 240}}}
```

已核实当前 validator 不支持递归对象/数组、联合 type、长度限制，且对 list 类型声明会抛 TypeError。B1 校验器必须显式检查 `type`（精确类型，bool 不当整数）、required、additionalProperties、enum、minimum、minLength/maxLength、maxItems/items；null 直接接受 None 后不再检查对象 required。当前工具 validator 不扩大修改；只读工具仍使用其现有标量协议。

规范规则只在去两端空白和末尾 `。.!！?？` 后 **fullmatch**：`取消|取消任务|取消本次任务` → CANCEL；`继续|继续任务` → RESUME；`(?:就去|选|选择)第([一二三四五六七八九十]|[1-9][0-9]?)个` → SELECT；`第([一二三四五六七八九十]|[1-9][0-9]?)个(?:多远|叫什么|多久能到)` → ASK_CANDIDATE。`好/可以/换一个/如果找不到就取消` 不命中这些规则。后两类含条件/替换对象不明必须 UNCLEAR；单独“好/可以”即使解释为 SELECT 也只可复用当前有效已选候选或唯一候选，不能确认。

源文本校验顺序：schema → intent/changes/reference 组合 → 每条非空 evidence 必须是本轮连续原文子串 → 明确字段/操作语义 → 规范化数值/引用。距离解析支持阿拉伯小数及中文整数 1–99，单位 `公里|千米|km|米|m`，米除 1000；仅接受唯一、正、有限距离值，`15 或 10 公里`、无单位、负号、范围歧义必须澄清。preferred_area 的 SET value 必须是 evidence 内原样文本；REMOVE 的 value 为 null，原文需含明确移除语义及该字段名（如“取消距离限制”“不限制距离”“取消区域偏好”）。unsupported SET value 为用户原文需求；unsupported REMOVE value 为原已存需求，evidence 为本轮明确放弃该需求的语句，reducer 再检查旧需求存在；“按已支持条件继续，放弃其他要求”可明确清除所有未解决条件，此时 REMOVE value 为 `"*"`，只允许这类明确放弃表达。保留未被移除的其他条件。SELECT/ASK 只允许一个 index 或 name；index 必须等于原文序数确定性解析结果，name 是原文子串。不能接受 JSON 列表代替单意图。

模型调用核心必须按以下顺序，不在异常后重试：

```python
started = time.perf_counter()
remaining = budget.claim_model()
response = llm.chat_with_timeout(
    messages, tools=[], timeout_seconds=min(timeout_seconds, remaining)
)
budget.remaining()
if response.tool_calls:
    raise ValueError("INTERPRETER_TOOL_CALL")
raw = json.loads(response.content or "")
error = validate_proposal(raw, text)
```

将 JSON/类型错误、超时、BudgetExceeded 分别转成 invalid `Interpretation`（`INVALID_PROPOSAL` 或精确验证码、`INTERPRETER_TIMEOUT`、预算原错误），保存 elapsed/usage/response_model。`tools=[]` 时即使客户端返回 tool calls 也拒绝，绝不执行。规则路径模型次数为 0；失败模型请求计数为 1。

- [ ] **Step 4 — 实现纯 reducer。** 按此完整决策表实现，不调用 `AgentTask.transition()`：

| 提议/状态 | 结果 |
|---|---|
| START + 当前业务未终止 | 当作显式条件/目标修订；不换 task ID，受 3 次上限约束 |
| START + IDLE/CANCELLED/COMPLETED/FAILED | operation=start、revision=0；由 coordinator 新建 task；不得从旧条件补新任务 |
| UPDATE + 活跃业务 | 在浅拷贝条件容器中合并已验证修改；每个有效值写本轮来源；变化一次只加一个 revision |
| UPDATE/RESUME/SELECT + 终止业务 | clarify `NEW_TASK_REQUIRED`，不复活 |
| 同值 SET / 删除不存在支持条件 | changed=False，不加 revision、不搜索、不延时；继续显示当前进度 |
| 第 4 次真实修订 | clarify `REVISION_LIMIT`；旧条件不改，coordinator 撤销旧 pending/选择执行资格，要求明确新任务 |
| 未支持条件仍存在 | clarify `UNSUPPORTED_CONSTRAINT`；有效支持条件可提交，但不搜索/选择/创建 pending；中文列出不支持需求 |
| 同句同字段矛盾修改、SELECT 混 changes | clarify `CONFLICTING_CONSTRAINTS`；整组不提交 |
| SELECT/ASK | 转 select/candidate_answer，reference 保留用户来源；有效期/映射由下一任务检查 |
| SIDE_QUESTION | side_question；原业务字段原样保留，changed=False |
| CANCEL | cancel；coordinator 终止待办，不调用 `cancel_navigation` |
| RESUME | resume；不清空未解决条件、不赋予授权 |
| UNCLEAR | clarify `AMBIGUOUS_REQUEST`；不更改条件/候选/pending |

澄清原因记录到对话 trace；invalid interpretation 不进入 reducer，原 task（包括 status/clarification）保持不变，独立候选过期处理除外。有效 UNCLEAR 可更新 task.clarification，但若当前 pending 仍有效，保留 AWAITING_CONFIRMATION 和 action 身份。

- [ ] **Step 5 — GREEN 与审查。** 同 Step 2 测试全通过；Sol 检查规则不过度匹配、源文本不等于授权、pure reducer 不变输入、schema 确实生效。交主线程，建议标题 `feat: validate dialogue intents and reduce sourced constraints`。

## Task 3：候选时效、业务协调与确认接入

**Files:** Create `modules/vehicle_ai/agent/dialogue_candidates.py`、`modules/vehicle_ai/agent/dialogue_coordinator.py`、`tests/vehicle_ai/test_dialogue_flow.py`; modify `modules/vehicle_ai/agent/vehicle_agent.py`、`modules/vehicle_ai/runtime.py`、`modules/vehicle_ai/agent/confirmation.py`、`modules/vehicle_ai/agent/plan_actions.py`、`modules/vehicle_ai/agent/plan_flow.py`、`modules/vehicle_ai/agent/session.py`; extend `tests/vehicle_ai/test_tool_confirmation.py`。

**Interfaces:**

- Runtime/VehicleAgent 新尾部参数 `dialogue_config: AgentDialogueConfig | None=None`；Runtime 另追加 `budget_clock: Callable[[], float]=time.monotonic` 并传 Agent。默认用 `AgentDialogueConfig()`，旧调用完全不启用 B1。
- Agent 属性 `dialogue_session: DialogueSession`、`dialogue: DialogueCoordinator`；reset 生成新 session，turn 清零；`chat` 只在 enabled 且 `dialogue.handles(text)` 时调用 `dialogue.chat(text, debug=debug)`。
- `DialogueCoordinator(agent, config)`；`handles(text: str) -> bool`：存在 B1 业务（含终止的旧业务以阻止“好/继续”复活）或 `TaskPlanFlow.is_rest_location_request(text)` 为真。初始非休息业务仍走旧路径；第一条休息请求必须解释，复杂句不能因命中关键词而自动执行。
- `search_candidates(agent, budget: TurnBudget) -> str`、`present_candidates(agent) -> str`、`invalidate_candidates(agent, reason: str) -> None`、`candidate_is_current(agent) -> bool`、`resolve_reference(agent, reference: dict | None) -> dict | None`、`select_candidate(agent, candidate: dict) -> str` 位于候选模块。
- `ActionConfirmationController.stage(..., expires_at: float | None=None)`；None 保持旧 TTL；新动作 TTL `min(120, expires_at - now)`，非正不建立；同参数旧 pending 只可保持或缩短，不能延长。由控制器构造新 PendingAction，调用者不直接 set 授权。

- [ ] **Step 1 — 写失败测试与可复用 fixture。** 用真实 Runtime/模拟导航工具/脚本解释器，不能 monkeypatch 整个 coordinator 绕过入口。建立 `tests/vehicle_ai/dialogue_support.py` 测试 helper：

```python
import json
from modules.config.agent_dialogue import AgentDialogueConfig
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime

def interpretation(intent, *, changes=(), reference=None, evidence="找附近服务区"):
    return ScriptedResponse(content=json.dumps({"intent": intent,
        "changes": list(changes), "reference": reference, "evidence": evidence,
        "clarification_reason": None}, ensure_ascii=False))

def make_runtime(*responses, clock=None, **kwargs):
    now = clock if clock is not None else [0.0]
    runtime = VehicleMindRuntime(ScriptedLLMClient(tuple(responses)),
        dialogue_config=AgentDialogueConfig(enabled=True),
        action_clock=lambda: now[0], quality_clock=lambda: now[0],
        enable_event_recommendations=False, **kwargs)
    return runtime, now

def test_select_second_requires_one_explicit_confirmation():
    runtime, now = make_runtime(interpretation("START"))
    runtime.chat("找附近服务区", debug=False)
    task = runtime.agent.task
    assert task.status.value == "AWAITING_INPUT"
    expected = task.plan.candidates[1]["poi_id"]
    runtime.chat("就去第二个", debug=False)
    pending = runtime.agent.pending_actions.get()
    assert pending.arguments["poi_id"] == expected
    assert pending.created_at + pending.expires_after_seconds <= task.plan.candidate_snapshot.expires_at
    assert runtime.context_manager.get_context().vehicle.navigation_destination_id is None
    assert runtime.agent.confirm_pending(pending.action_id).success
    assert not runtime.agent.confirm_pending(pending.action_id).success
    writes = [x for x in runtime.tools.execution_history()
              if x.name == "start_navigation" and x.confirmed and x.success]
    assert len(writes) == 1
    assert runtime.agent.task.task_id == task.task_id
```

再加入：更新后旧 action_id 永失效；搜索失败不回退旧候选；120 秒等号边界；从未展示/序号越界不猜；名称只匹配当前候选规范名/别名（复用 `resolve_target`）；多候选“换一个”澄清；取消后“好/继续”零写入；第 4 次更新无新搜索；同值条件不重建；expiry refresh 消耗同一计划步数，不能靠持续“继续”重建预算；reset 与双实例无串话；默认关闭旧 golden 路径不调用解释器。确认前 expires_at 已过时必须失败，即使 chat 未再运行。

- [ ] **Step 2 — RED。** `uv run --group dev python -m pytest tests/vehicle_ai/test_dialogue_flow.py tests/vehicle_ai/test_tool_confirmation.py -q`；新入口/参数缺失失败。
- [ ] **Step 3 — 实现候选契约与原计划衔接。** `search_candidates` 从 task.constraints 构造真实搜索参数，省略被删除的键，不让模型决定参数。`budget.claim("search_nearby_rest_area", args)` → `plan_flow.before_tool()` → `tool_registry.execute(..., user_intent=current_user_intent)` → 原 `observe_tool()`；工具结果写 task/trace，失败保留新条件、清候选执行资格。仅明确 retryable 且 `budget.claim_retry()` 与 `plan_flow.claim_read_retry()` 均允许时执行一次原重试，仍用 `observe_tool(...allow_retry=True/False)` 正确结束 SEARCH 步骤。

成功后在原 plan 上生成唯一新快照，`created_at=agent.pending_actions.now()`、`expires_at=created_at+ttl`、`constraint_revision=task.revision`；展示必须由系统直接遍历 `plan.candidates` 生成序号/名称/距离/预计时间，不通过 LLM 重排。只有实际返回该列表时才写 `presented_turn_id`、`presented_poi_ids` 与 task.last_presented_candidate_set_id。

当前有效的完整判定为：task 非终止、plan RUNNING、snapshot 存在、snapshot.constraint_revision==task.revision、now<expires_at、snapshot.candidate_set_id==task.last_presented_candidate_set_id、presented_turn_id 非空、映射中每个 ID 在当前 candidates 恰好一个。序号索引只能用 `presented_poi_ids[index-1]` 再回查 candidates。名字调用已有 `resolve_target`，任何歧义返回 None。最后展示映射不能为空，不因为当前 candidates 存在就擅自建立序号。

`invalidate_candidates` 清空候选、selected_poi_id、snapshot、last_presented ID 与 store pending；同步 task.pending_action=None。若当前步骤为 AWAIT_CONFIRMATION，用原 `_finish_step(...success=False,error=reason)` 关闭，expiry 保持 plan RUNNING 后 AWAITING_INPUT；条件改变时调用原 `plan_flow.cancel(...task_status=AWAITING_INPUT)` 终止旧 plan。候选过期只撤销资格，不重新保存旧 PendingAction。

`select_candidate` 先验证快照及 unresolved 为空，复用 `TaskPlanFlow._stage_candidate()`，不能通过工具 execute 导航；pending metadata 含 `candidate_set_id`、`constraint_revision`、`expires_at` 和规范候选来源。`PlanActionsMixin` 增加薄方法 `_candidate_stage_options(agent, plan) -> dict`，B1 下返回版本 metadata 和 expires_at，旧模式返回空；现有 `_stage_candidate` 与 `_propose_alternative` 调用前各取 `options`，将 `options.get("metadata", {})` 合并入原 metadata 字典，并显式传 `expires_at=options.get("expires_at")`，不能同时传 metadata 和包含同名键的 `**options`。确保恢复候选也受原快照到期约束。保持原选择/确认/执行/验证步骤，不复制计划器。

确认入口在旧 `begin_confirmation` 之前调用候选模块的 `validate_pending_candidate(agent, action_id: str) -> bool`：仅对启用 B1 的导航 pending，检验快照/版本/选择 ID/pending metadata；失败清 store/同步快照/转 AWAITING_INPUT，并返回 `ToolResult(False, "候选已失效，请重新查询后选择。", error="INVALID_CONFIRMATION")`。通过后仍交原 `confirm_pending()` 完成签发、执行、验证。旧 action ID 不因当前另一个有效 pending 而误清新动作；ID 不匹配直接交原 invalid confirmation 路径。取消/拒绝终止任务后旧候选即使残留报告快照也无执行资格。

- [ ] **Step 4 — 实现协调器入口。** chat 先 strip；空输入不增轮。每个 B1 非空回合 `next_turn()` 并设置 current_user_intent、重置 music 提示；建立一次 `TurnBudget(min(90, agent.turn_timeout_seconds), min(10, agent.max_tool_calls), agent.budget_clock, max_model_calls=min(5, agent.max_tool_rounds))`。独立处理自然过期，再记录 user_request。coordinator 按 Task2 的精确六键契约构造 summary：条件来源用 ConstraintValue 的三个字段；候选须先通过 candidate_is_current，再按 snapshot.presented_poi_ids 回查 TaskPlan.candidates，以 enumerate(..., start=1) 生成 index，以 plan_flow.display_name(candidate) 生成 name，并复制其字符串 aliases。不得传完整 task.to_dict()/候选dict或额外 ID，不生成别名字段；无有效展示则 candidates=[]。解释→验证→reducer→一次提交；无效解释记录澄清回复和单次问答错误，不写 task。业务操作按下表接入：

| operation | 协调器提交与回复 |
|---|---|
| start | 新 `AgentTask(goal=text)`，revision 0，提交条件；复用 `plan_flow.begin(agent, goal)`；未解决条件先澄清，否则搜索/展示 |
| search | 保持 task ID；先撤销旧候选与 pending，旧 plan cancel；提交新条件/revision，复用 begin 生成新 plan；失败不回滚 |
| select | resolve_reference；失败澄清；成功仅 stage；相同当前候选已有有效 pending 时保留其身份，不再加 SELECT 步骤 |
| candidate_answer | 从有效候选返回名称/距离/工具实际 eta 字段；缺字段直说未提供，设施不推断；不建 pending、不加计划步骤 |
| side_question | Task 4 接入只读循环；Task 3 验证时先覆盖 reducer 保持不变，不能发布一个假装已回答的分支 |
| resume | 有未解决条件先澄清；当前候选有效则重新展示而不续期；已过期仅在原 plan 剩余步数允许时再次 SEARCH；终止 task 要求新请求 |
| cancel | 清 pending/候选资格；原 plan cancel，task=CANCELLED；不调用正在运行导航的取消工具 |
| clarify | 输出明确原因；若 revision 超限，撤销执行资格并 AWAITING_INPUT；其他不撤销仍有效 pending；支持条件已接受则记录来源与差异 |

若用户在 revision 超限后明确“重新开始找服务区”可新建 task：原 task.reason 为 REVISION_LIMIT，验证后提议为 START，且本轮 fullmatch `重新开始(?:找|查询)(?:附近)?(?:服务区|休息地点)` 时先终止旧任务再提交新 START；仅“继续”“改成…”不能借此逃脱上限。搜索后的业务状态是 AWAITING_INPUT，不调用 `task.finish()`。异常预算/PlanTransitionError 用中文停止本轮并记录原因；不重建预算，不回滚已接受条件，不把计划失败伪装成功。每个答案恰好一次 `_record_final_response()`，历史仍截断 12 条，结构化条件不依赖历史。

`session.record()` 仅在 B1 enabled 时补 `session_id/turn_id/task_id/task_revision`；新 `dialogue_interpretation` 事件含提议、valid、reason、source、usage、elapsed；`dialogue_transition` 含 operation、reason、字段级 before/after、candidate_set_id；`dialogue_budget` 含 model_calls/tool_calls/remaining（到期记 0）。沿用最多 200 trace 事件，不输出整模型隐藏推理。

- [ ] **Step 5 — GREEN 与审查。** 运行新 flow、原 plan_flow、pending、target_resolution、confirmation、navigation 全套：`uv run --group dev python -m pytest tests/vehicle_ai/test_dialogue_flow.py tests/vehicle_ai/test_agent_plan_flow.py tests/vehicle_ai/test_tool_confirmation.py tests/vehicle_ai/test_pending_intent.py tests/vehicle_ai/test_target_resolution.py tests/vehicle_ai/test_navigation_candidates.py -q`。再运行 `uv run --group dev python scripts/check_source_size.py`。Sol 重点审 action 生命周期与 default-off；交主线程，建议标题 `feat: coordinate versioned rest-stop dialogue and confirmation`。

## Task 4：只读插入问答、共享预算与状态保持

**Files:** Create `modules/vehicle_ai/agent/dialogue_readonly.py`、`tests/vehicle_ai/test_dialogue_readonly.py`; complete `modules/vehicle_ai/agent/dialogue_coordinator.py`; extend `tests/vehicle_ai/test_dialogue_flow.py`。

**Interfaces:** `answer_readonly(agent, text: str, budget: TurnBudget, *, debug: bool=False) -> str`。允许集合精确 `get_vehicle_status get_climate_status get_media_status search_vehicle_knowledge query_trip_events` 与 registry 实际存在且 `read_only=True` 的交集；不含地点搜索（防止暗中替换业务候选）。复用 SYSTEM_PROMPT、`agent._context_message()` 的当前质量检查、`trip_memory_message()`，不把历史问答当当前观测。

- [ ] **Step 1 — 写失败测试。** 必测插入前后 task ID/constraints/revision/plan steps/selected ID/pending action ID/expiry 相等；问答失败不将业务 FAILED；模型请求写工具全部被执行层阻止；一批混合只读+写调用均有正确 tool result 配对；只读错误也不占计划步骤；过期发生在模型回答期间时 pending 清理后不得恢复；超过 12 条历史仍保留最早条件；模型解释占 1 次后回答最多 4 次；时间预算只剩 2 秒时解释超时≤2、后续不能续 90。

```python
from copy import deepcopy
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from dialogue_support import interpretation, make_runtime

def test_side_question_denies_write_without_touching_business():
    runtime, now = make_runtime(interpretation("START"),
        interpretation("SIDE_QUESTION", evidence="当前空调状态如何"),
        ScriptedResponse(content=None, tool_calls=(LLMToolCall(
            "bad", "set_temperature", {"temperature_c": 23},
            '{"temperature_c":23}'),)),
        ScriptedResponse(content="本次只读问答不能修改空调。"))
    runtime.chat("找附近服务区", debug=False)
    runtime.chat("就去第二个", debug=False)
    before = deepcopy(runtime.agent.task.to_dict())
    action = runtime.agent.pending_actions.get()
    runtime.chat("当前空调状态如何", debug=False)
    after = runtime.agent.task.to_dict()
    for key in ("task_id", "constraints", "revision", "plan", "status"):
        assert after[key] == before[key]
    assert runtime.agent.pending_actions.get().action_id == action.action_id
    assert not any(x.name == "set_temperature" for x in runtime.tools.execution_history())
    assert any(x["kind"] == "dialogue_readonly_result" for x in runtime.agent.trace)
```

仓库 tests 没有 `__init__.py`；同目录测试使用上例 `from dialogue_support import ...`，由现有 pytest 默认 prepend 导入模式定位 helper，不新增包或全局 sys.path 改动。

- [ ] **Step 2 — RED。** `uv run --group dev python -m pytest tests/vehicle_ai/test_dialogue_readonly.py -q`；未接入只读回答失败。
- [ ] **Step 3 — 实现允许集合和循环。** 模型 schemas 仅发允许工具，执行前仍做允许集合 **及** `definition.read_only` 双重检查；任何未知/写工具生成 `READ_ONLY_TOOL_DENIED`，不调用 registry、不 stage、不调用 plan_flow。参数沿用 JSON raw==arguments 和 `valid_arguments` 校验；允许调用先 `budget.claim()` 再 registry.execute；这里不自动工具重试。每次 model 调用前 `budget.claim_model()`，只用其返回 remaining timeout；成功后再 `remaining()`。失效/模型异常/空回复以单次问答失败 trace 结束，原业务 status/reason 不变（自然过期可转 AWAITING_INPUT）。

执行门核心代码：

```python
READ_ONLY_NAMES = frozenset({"get_vehicle_status", "get_climate_status",
    "get_media_status", "search_vehicle_knowledge", "query_trip_events"})

def execute_readonly(agent, name, arguments, budget):
    if name not in READ_ONLY_NAMES or name not in agent.tool_registry.names():
        return ToolResult(False, "本轮只允许读取状态与证据。", error="READ_ONLY_TOOL_DENIED")
    definition = agent.tool_registry.get(name)
    if not definition.read_only:
        return ToolResult(False, "本轮只允许读取状态与证据。", error="READ_ONLY_TOOL_DENIED")
    if not valid_arguments(arguments, definition.parameters):
        return ToolResult(False, "只读工具参数不符合要求。", error="INVALID_ARGUMENTS")
    budget.claim(name, arguments)
    return agent.tool_registry.execute(name, arguments, user_intent=agent.current_user_intent)
```

只读工具结果仅写 `dialogue_readonly_result` trace，不写 task.last_tool_result/tool_results/reconciliation/plan。循环每批 assistant tool calls 后逐一添加对应 call_id 的 tool result，包括拒绝；遇预算终止也不能把不完整调用消息写入 history（history 只存最终问答文本）。最终 coordinator 记录一次 agent_reply；只读 runner 不自行记录 history。结果 trace 包含 `answer_status=success|failed`，与业务 TaskStatus 分离，snapshot 只作测试不变性比较，代码无 `pending_actions.set(old_pending)` 或 `agent.task=old_snapshot`。

前后都执行自然到期检查。若纯问答期间候选过期，撤销 pending、保留条件和 task ID，计划内 AWAIT_CONFIRMATION 步骤关闭；后续 RESUME 在原计划剩余预算内重新搜索。不能从 snapshot 还原旧 action，也不能把未过期 action 重新 stage 延时。

- [ ] **Step 4 — GREEN 与完整回归。** `uv run --group dev python -m pytest tests/vehicle_ai/test_dialogue_readonly.py tests/vehicle_ai/test_dialogue_flow.py tests/vehicle_ai/test_context_quality.py tests/vehicle_ai/test_prompt_injection_safety.py -q`。再跑 `uv run --group dev python -m pytest -m "not hardware and not online" -q`；基线为 725 passed、2 deselected，新测试增加后全部通过且旧题数不变。
- [ ] **Step 5 — 审查交付。** Sol 审共享预算、无业务状态写入、超时后清理、恶意工具请求与 pending 不复活；交主线程，建议标题 `feat: answer side questions under shared read-only budgets`。

## Task 5：冻结 D01–D16、可重复运行器与三指标

**Files:** Create `scenarios/agent_eval/dialogue/cases.yaml`、`scenarios/agent_eval/dialogue/scripted.yaml`、`scenarios/agent_eval/dialogue/manifest.yaml`、`modules/vehicle_ai/evaluation/dialogue_cases.py`、`modules/vehicle_ai/evaluation/dialogue_runner.py`、`modules/vehicle_ai/evaluation/dialogue_metrics.py`、`tests/vehicle_ai/evaluation/test_dialogue_cases.py`、`tests/vehicle_ai/evaluation/test_dialogue_runner.py`、`tests/vehicle_ai/evaluation/test_dialogue_metrics.py`。

同任务增加 `modules/vehicle_ai/evaluation/dialogue_review.py` 与 `tests/vehicle_ai/evaluation/test_dialogue_review.py`，职责为下述审核包导出/验证/合并，不调用额外在线评审模型。

**Interfaces:**

- `load_dialogue_cases(root: Path) -> tuple[dict, ...]`：精确 D01–D16、完整 schema 和文件 SHA-256，任何改变先拒绝运行。新集合独立 schema，不扩展旧 EvaluationCase 强行塞入新字段。
- `run_dialogue_case(case: dict, client_factory: Callable[[], BaseLLMClient], *, provider: str, model: str, repetition: int) -> dict`：一个场景可按 `session` 建立隔离 runtime，用可注入 action clock；budget clock 默认真实 monotonic，故障测试用专门 budget advance，二者不可混用。
- `grade_dialogue_trial(case: dict, trial: dict) -> dict`；`summarize_dialogue_trials(trials: list[dict]) -> dict`。结果保留 state_checks/task_checks/clarification_checks 每条 pass/fail/observed/expected，不以答案含一个关键词判整题成功。
- `export_review_packets(run: Path, output: Path) -> None`；`apply_dialogue_reviews(run: Path, decisions: Path, output: Path) -> dict`；同文件 CLI `python -m modules.vehicle_ai.evaluation.dialogue_review export|apply --run PATH --output PATH`，apply 另需 `--decisions PATH`。输出存在则拒绝覆盖。

- [ ] **Step 1 — 写失败测试。** 必测少一题/多一题/错 hash/修改预期拒绝；执行同一冻结场景两次 ID 不串；mock provider 故障保留 trial 与分母；一个任务失败导致 TSR fail、部分状态失败产生分子/分母、澄清零适用项为 N/A 非 100%；缺失 check 为 failed；未完成语义审查不能被计为已核验成功。

```python
from pathlib import Path
from modules.vehicle_ai.evaluation.dialogue_cases import load_dialogue_cases
from modules.vehicle_ai.evaluation.dialogue_metrics import summarize_dialogue_trials

def test_frozen_dialogue_suite_and_denominators():
    cases = load_dialogue_cases(Path("scenarios/agent_eval/dialogue"))
    assert [case["id"] for case in cases] == [f"D{i:02d}" for i in range(1, 17)]
    summary = summarize_dialogue_trials([
        {"case_id": "D01", "task_success": True, "state_checks": [True, True],
         "clarification_checks": [], "review_status": "ai_reviewed", "error": None},
        {"case_id": "D12", "task_success": False, "state_checks": [True, False],
         "clarification_checks": [False], "review_status": "ai_reviewed", "error": "MODEL_TIMEOUT"}])
    assert summary["task_success_rate"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary["state_consistency"] == {"passed": 3, "total": 4, "rate": 0.75}
    assert summary["clarification_correctness"] == {"passed": 0, "total": 1, "rate": 0.0}
```

- [ ] **Step 2 — RED。** `uv run --group dev python -m pytest tests/vehicle_ai/evaluation/test_dialogue_cases.py tests/vehicle_ai/evaluation/test_dialogue_runner.py tests/vehicle_ai/evaluation/test_dialogue_metrics.py -q`；新模块缺失失败。
- [ ] **Step 3 — 落盘完整 16 条固定会话和逐步断言。** cases.yaml 顶层 `schema_version: 1`、`mode: dialogue_b1`、`review_status: ai_reviewed`、`poi_catalog`、`cases`；每 case 精确字段 `id/title/synthetic/fault_injected/steps/task_checks/state_checks/clarification_checks/semantic_checks`。step 精确允许 `session`（默认 main）、`at_seconds`、`user_text`、`confirm_pending`、`reject_pending`、`remember_pending_as`、`confirm_saved`、`tool_failure`、`model_fault`、`budget_override`、`cabin_quality`、`vehicle`；检查只用白名单 `eq/ne/same_as/is_none/count_le/contains`，禁止 eval/任意 Python 或任意 dotted getattr。

每步运行保存 reply、session/turn/task、完整 task 序列化、当前 pending、安全 vehicle 状态、该步工具记录、新 trace、模型请求/响应 usage 与耗时。check 结构固定 `{step: int, path: str, op: str, expected: value}`；`same_as` 的 expected 为 `{step: int, path: str}`；path 只能 JSON 字典/列表导航。`confirm_saved` 验证旧 action 不可用。model_fault 仅作用于指定步骤，记录 `injected=true`，不能伪装模型自然产生的结果。

固定 POI 使用现有默认目录的两个地点及全部真实字段（`rest_area_001` 为6.8公里/8分钟，`rest_area_002` 为12.4公里/15分钟，从 `navigation.py` 复制到新冻结集，并在 manifest 绑定），不可为了提高通过率添加第三个地点或修改工具事实。完整会话内容和核心断言如下，写成 YAML 时一个动作一 step：

| ID | 固定会话/注入 | 冻结核心断言 |
|---|---|---|
| D01 | 找附近服务区 → 十五公里以内 | task 相同；revision=1；来源 turn=2；最后搜索 max_distance_km=15 |
| D02 | 找附近服务区，十五公里以内 → 就去第二个 → 保存 old → 改成十公里以内 → 确认 old | revision=1；old 无效；新搜索=10；旧候选 set ID 不同；零成功导航 |
| D03 | 找西湖附近十五公里以内的服务区 → 取消距离限制，仍保留西湖偏好 | 删除距离；preferred_area=西湖；后搜索无距离键；task 相同 |
| D04 | 找十五公里以内有厕所的服务区 → 就去第一个 → 放弃厕所要求，按十五公里以内继续 | 首轮保留未解决厕所、无 pending；第二轮不能选；明确放弃后搜索=15 |
| D05 | 找附近服务区 → 第二个多远 | 回复名称/距离对应实际展示第二个；无 pending、无导航、无新增 SEARCH |
| D06 | 找十五公里以内的服务区 → 当前为什么提示我疲劳 → 继续 | 注入 cabin_quality=invalid；回答不把未知风险说成当前确定事实；task/条件相同，plan 不被问答完成 |
| D07 | 找附近服务区 → 就去第二个 → 显式 confirm_pending → 再 confirm_saved | 确认前零导航成功；确认后精确第二个一次；重放失败；任务 COMPLETED |
| D08 | 找附近服务区 → 换一个 | 多候选澄清；无默认选第二个；候选顺序不变 |
| D09 | 找附近服务区 → 就去第二个，保存 old → action clock=120 → 继续 → confirm old → 就去第二个 | old 无效；新搜索；新 set 不同；新 pending ID 不同且 TTL≤新候选 TTL |
| D10 | 找附近服务区 → 就去第二个 → 取消 → 好 → 继续 | CANCELLED 不复活；零导航；旧 pending 永无效 |
| D11 | 找附近服务区 → 就去第二个 → 下一次搜索注入 TRANSIENT_ERROR 且 retryable=false → 改成十公里以内 → 好 | 新条件=10 保留；旧候选/pending 不可执行；搜索失败明确；零导航 |
| D12 | main/invalid/conflict 三个独立 session 各先找服务区；随后分别“十五公里以内”注入 TimeoutError、“十五公里以内”注入非法 JSON、“十五公里或十公里都行”正常在线解释 | 各原业务条件/候选不被错误解释改写；前两者一次解释失败，第三者澄清；注入和真实解释分开标记 |
| D13 | 找附近服务区 → 就去第二个 → 当前空调状态如何（回答阶段注入 set_temperature 工具请求） | 写工具未执行；task/选择/pending 不变；trace 明确拒绝；不算模型自然攻击成功率 |
| D14 | A 找西湖附近服务区；B 找河滨附近服务区；A 就去第一个；B 取消；A confirm_pending | 不同 session/task/candidate/action；B 不影响 A；A 导航只到 A 展示目标 |
| D15 | 找十五公里以内服务区 → 连续七轮“当前车辆状态如何” → 继续 | history≤12；最早用户句已截断；结构化距离仍15、来源仍首轮，恢复不靠历史原句 |
| D16 | revision session：找服务区→20公里→15公里→10公里→5公里；budget session：找服务区→只读问答阶段注入连续只读模型调用请求，max_model_calls=2；tool session：同类问答，max_tool_calls=1 | 第4次变化拒绝；revision≤3；解释+回答共享模型上限；工具≤1；无重新启动预算 |

候选 expiry 专门补充确定性子测试：`119.999` 有效、`120.0` 无效、未展示索引、序号 99、在一次问答模型调用期间时钟推进。D12/D13/D16 的注入包装器只替换明确标注的响应，其余步骤仍调用该 provider；报告独立显示注入覆盖，不能把这些当作自由模型准确率。

冻结顺序：先将16场景的有限脚本响应写入 scripted.yaml，全部跑通，Sol 审断言和语义 rubric，再生成 manifest。manifest 绑定 cases.yaml 与 scripted.yaml 各自原始 bytes SHA-256、16 IDs、schema_version、mode、冻结日期、synthetic/ai_reviewed 标签；之后在线启动不得自改题/预期/hash。改题必须新版本并保留旧结果，不回写本版。

- [ ] **Step 4 — 实现 runner 与 metrics。** 复用 `evaluation.runner.RecordingClient` 的超时/usage 捕获，但 B1 请求分类由本轮 trace phase（interpretation/read_only）判断，不能沿用“tools=[] 就是 event_advice”误标签。可在 B1 runner 的记录包装器增加 `phase`，不改变旧评测定义。在线记录 case hash、manifest hash、source revision/tree dirty、agent/dialogue config hash、provider/model/requested and actual model、温度0.2、超时、重试0、session隔离、故障注入列表。

Task Success Rate 分子为全部 task_checks、必要 state_checks 和 semantic_checks 已审通过的 trial，分母为全部安排的 trial（含模型/工具错误）。未审 semantic 项标 needs_review，不计通过；可另报自动检查通过率。State Consistency 按冻结 state_checks 总项数计算，缺失的后续步骤检查判失败；Clarification Correctness 按冻结 clarification_checks 项数，包含应澄清和不应无端澄清的轮次，故障后缺失判失败。语义检查分开保留 reviewer/source/status/结论与理由，AI 审核可更新但保留审计记录；不新增“人工已审”假标签。

指标计算公共片段：

```python
def ratio(values: list[bool]) -> dict:
    passed, total = sum(value is True for value in values), len(values)
    return {"passed": passed, "total": total,
            "rate": passed / total if total else None}
```

输入错误/网络失败生成原计划的 trial 结果，不 silently skip。保存每模型每场景3次分项，延迟报告 p50/p95（明示取样方法）、总 tokens 和缺失 usage 数；无法获得 usage 不记为0。失败枚举含解释失败、澄清错误、状态漂移、授权误用、预算超限、工具/模型基础设施失败。

语义审核使用结构化文件，不依赖口头“已看过”。每个 semantic_check 冻结 `{id, step, requirement}`；export 读取 run/result.json，为全部48 trials导出一个 JSON 数组，条目包含 `trial_id/case_id/repetition/case_sha256/trial_sha256/semantic_checks/steps/replies/source_facts/tool_results/mechanical_checks`，trial_id 是 `{provider}:{case_id}:{repetition}`，sha256来自该trial规范JSON。审核者 Sol 逐条阅读并用 apply_patch 写 decisions.json：顶层精确 `schema_version:1/reviewer:"GPT-6 Sol High AI review"/independent_human_review:false/run_sha256/decisions`；每decision精确 `trial_id/trial_sha256/items`；items 覆盖每个 semantic_check ID，值精确 `{status:"pass"|"fail"|"unclear", evidence:"非空的实际轮次与输出依据"}`。错误trial也必须有decision，证据说明缺失或失败，不能跳过。

apply 验证源run/每trial hash、48唯一trial精确覆盖、item ID精确覆盖、非空证据；拒绝缺项、重复、跨run/过期hash和把mechanical fail覆盖为pass。`unclear`→needs_review且TSR不计通过；语义fail→task失败；只有所有mechanical与semantic通过才task_success=True。状态/澄清机械断言原结果不因审核改变。输出独立 `reviewed.json`（含全部原trial、decisions、review_status、重算summary、source run/decision哈希）和 `reviewed.md`，不改result.json。二次审计输出新文件并含旧审核文件hash、逐项旧判/新判/证据，不覆盖先前审核。测试固定覆盖缺一trial、重复trial、少一semantic item、hash错、机械失败不可翻转、unclear保留分母与审计链。

- [ ] **Step 5 — GREEN 与审查。** Step 2 测试通过；确定性运行16题全部 state/task/clarification checks通过。Sol审 D01–D16 完整覆盖、冻结 hash、判分分母、故障注入标签。交主线程，建议标题 `test: freeze and grade sixteen dialogue scenarios`。

## Task 6：中文演示、双模型各三次在线验证与结项

**Files:** Create `modules/vehicle_ai/evaluation/dialogue_cli.py`、`modules/vehicle_ai/evaluation/dialogue_report.py`、`tests/vehicle_ai/evaluation/test_dialogue_cli.py`、`tests/vehicle_ai/evaluation/test_dialogue_report.py`、`docs/guide/agent-dialogue.md`、`docs/reports/2026-09-27-agent-dialogue-business-state.md`; update `README.md`、`todolist.md`。原始结果仅写 `runs/agent_dialogue/`，不提交密钥/大体积原始运行目录。

**Interfaces:** `python -m modules.vehicle_ai.evaluation.dialogue_cli`；参数 `--provider {scripted,qwen,glm}`（required）、`--model`、`--cases`（默认 `scenarios/agent_eval/dialogue`）、`--repetitions`（默认3，必须正整数）、`--output-root`（默认 `runs/agent_dialogue`）、`--run-name`（可选，限字母数字及连字符；默认时间戳；目标存在拒绝覆盖）、`--case-id` 可重复、`--temperature`（默认0.2）、`--timeout-seconds`（默认30）。入口强制 B1 enabled 并保存实际配置，不改变仓库默认 false。`render_dialogue_report(result: dict, destination: Path) -> None` 生成 report.html/report.md/result.json。

- [ ] **Step 1 — 写失败测试。** CLI dry scripted 模式不访问网络/密钥；hash错退出非零；结果96条按模型48+48展示；HTML 对用户文本/模型文本/工具字段使用 `html.escape`；原 JSON 在 `<details>`；中文目标/条件/未解决条件/候选/待确认/原因/指标齐全；没有把独立 GIF 说成当前运行视频证据；不将 raw key、provider client repr 或 env 放 provenance。

```python
from modules.vehicle_ai.evaluation.dialogue_report import render_dialogue_report

def test_report_escapes_user_text_and_shows_b1_mode(tmp_path):
    result = {"mode": "dialogue_b1", "provider": "scripted", "model": "scripted",
              "case_count": 16, "repetitions": 1, "trials": [], "summary": {},
              "notes": ["<script>alert(1)</script>"]}
    render_dialogue_report(result, tmp_path)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "业务状态与多轮对话" in html
    assert "dialogue_b1" in html
    assert "<script>alert(1)</script>" not in html
    assert "<details" in html
```

- [ ] **Step 2 — RED。** `uv run --group dev python -m pytest tests/vehicle_ai/evaluation/test_dialogue_cli.py tests/vehicle_ai/evaluation/test_dialogue_report.py -q`；入口/渲染未实现失败。
- [ ] **Step 3 — 实现报告和中文自测入口。** 首屏显示固定16独立场景、重复数、模式、模型、TSR/State Consistency/Clarification Correctness 分子分母、needs_review与失败数；逐场景逐轮展示本轮话语→意图/是否验证→条件增删→候选展示与失效原因→pending是否仅待确认→真实工具执行结果。raw prompt/trace/provenance折叠，指标 None 显示“不适用/未获得”。GIF 复用 `assets/demo/cabin_demo.gif` 和 `assets/demo/driving_perception.gif`，用 `os.path.relpath` 从报告位置引用，并注明“独立感知演示素材，并非本次对话同步视频”。不重制 GIF。

`scripted` 使用 Task 5 冻结场景对应有限脚本响应，响应映射放 `scenarios/agent_eval/dialogue/scripted.yaml` 并由 manifest 一并绑定；这是离线测试替身，不称端侧模型。默认 online provider 模型沿用环境中的 QWEN_MODEL/GLM_MODEL 与已有 build_llm_client 规则，不在代码内编造当前模型版本。只打印 provider/model、输出路径和完成数量，不打印配置密钥。报告仅使用允许的安全配置字段。

- [ ] **Step 4 — GREEN、完整工程检查、离线演示。**

```bash
uv run --group dev python -m pytest -m "not hardware and not online" -q
uv run --group dev ruff check apps modules scripts tests
uv run --group dev ruff format --check apps modules scripts tests
uv run --group dev mypy modules/config modules/vehicle_ai/context/enums.py modules/vehicle_ai/integration/cabin_adapter.py scripts/verify_assets.py
uv run --group dev python scripts/check_source_size.py
uv run --group dev python scripts/verify_assets.py --schema-only
uv run --group dev python -m modules.vehicle_ai.evaluation.dialogue_cli --provider scripted --repetitions 1 --output-root runs/agent_dialogue/offline
```

预期工程检查全部0退出、离线16题全部确定性断言通过。新生产文件逐一检查≤300行；超过先审职责再决策，不能移动任意尾部代码只为过线。主线程先提交代码/冻结集后再正式在线运行，使 provenance 能绑定实现 commit；用户保留的 `:memory:.ses` 导致 tree dirty 时如实记录并说明与代码无关，不删除它伪造 clean。

- [ ] **Step 5 — 正式在线运行与逐条审核。** 用户已授权在线验证，执行如下两批，每题各3次；可并行独立 provider 进程但每个 runtime 内仍顺序，不自动重试失败 trial：

```bash
uv run --group dev python -m modules.vehicle_ai.evaluation.dialogue_cli --provider qwen --repetitions 3 --output-root runs/agent_dialogue/online --run-name b1-qwen
uv run --group dev python -m modules.vehicle_ai.evaluation.dialogue_cli --provider glm --repetitions 3 --output-root runs/agent_dialogue/online --run-name b1-glm
```

预期产生48+48计划trial，运行失败仍落盘。检查16独立ID各出现3次、hash与配置相同、实际模型名、解释调用无工具、预算上限，审核96条多轮语义检查，保留“AI辅助审核”标记；发现自动审核误判要记录旧判/新判/证据，重算分数，不改题。在线成绩不要求100%；模型语义错误作为结果保留，不能因此无限调提示/改代码/重测。只有可由确定性失败测试证明的实现缺陷才修复，记录新版本并做一次受影响验证；原正式批次仍单独完整报告，不混选最好轮次。若 provider 暂不可达/权限不足，完成另一模型和所有离线任务，把缺失48条明确记为未运行，向主线程报告真实外部阻碍，不能声称双模型验证完成。

导出可审阅证据包的实际命令：

```bash
uv run --group dev python -m modules.vehicle_ai.evaluation.dialogue_review export --run runs/agent_dialogue/online/b1-qwen --output runs/agent_dialogue/online/b1-qwen/review-packets.json
uv run --group dev python -m modules.vehicle_ai.evaluation.dialogue_review export --run runs/agent_dialogue/online/b1-glm --output runs/agent_dialogue/online/b1-glm/review-packets.json
```

主线程将两个包交Sol按上述契约逐trial审查，分别保存各run目录的 `decisions.json`，不能由实现者基于mechanical pass自动填写语义pass。Sol完成后Luna执行：

```bash
uv run --group dev python -m modules.vehicle_ai.evaluation.dialogue_review apply --run runs/agent_dialogue/online/b1-qwen --decisions runs/agent_dialogue/online/b1-qwen/decisions.json --output runs/agent_dialogue/online/b1-qwen/reviewed.json
uv run --group dev python -m modules.vehicle_ai.evaluation.dialogue_review apply --run runs/agent_dialogue/online/b1-glm --decisions runs/agent_dialogue/online/b1-glm/decisions.json --output runs/agent_dialogue/online/b1-glm/reviewed.json
```

apply 同步调用报告渲染器在run的 `reviewed-report/` 子目录生成审核后HTML/MD/JSON（保留原run与原报告），最终文档只引用这两个reviewed.json汇总值，并链接全部原始失败和审查依据。

- [ ] **Step 6 — 文档、视觉验收与最终审查。** 打开实际生成的 HTML 检查中文字段、表格/折叠、长文本、GIF相对路径及失败状态，保存检查证据；不能只用字符串测试宣称视觉完成。`docs/guide/agent-dialogue.md` 写启用/默认关闭、完整演示命令、一次确认、未支持条件、到期/预算、只读边界、状态字段和故障含义。README 加“B1 多轮业务自测”链接与实测范围；todolist B1 只有确定性验收和在线审核实际完成后勾选。报告写实际 commit/hash、测试计数、每模型48 trials与16独立场景、三指标/延迟/tokens/失败/AI审核修订；在线数据缺失则明确未完成，不填零/虚构成绩。Sol 进行终审，主线程提交文档与代码，不提交 runs，建议标题 `docs: publish verified dialogue demonstration and evaluation`。

## 最终验收与计划自检

- [ ] 设计 §3–4 对应 Task1/3：唯一状态源、条件来源、快照版本/时效、3次修订、9步/1恢复、生命周期。
- [ ] 设计 §5 对应 Task2/4：固定 schema、原文核验、一次解释、15秒、90秒/5模型/10工具共享。
- [ ] 设计 §6 对应 Task3/4：只读隔离、无旧pending恢复、取消/完成不复活、双实例隔离、自然过期。
- [ ] 设计 §7 对应 Task1/6：默认关闭旧入口兼容、中文展示、真实trace、README与独立GIF范围。
- [ ] 设计 §8 全部D01–D16对应 Task5；Task6 双模型各3次，冻结集分母独立，结果真实且审核可追溯。
- [ ] 实施前/后确认 `git diff -- scenarios/agent_eval/golden` 为空；不更新旧40题hash或预期；`:memory:.ses` 原样保留。
- [ ] 本计划为执行协议，未声称生产实现或测试完成。主线程按 Task1→6交 Luna实施，每任务通过 Sol审查再进入下一项；Task3 的 side_question 接口在 Task4 完成前不得当作完整 B1 发布。

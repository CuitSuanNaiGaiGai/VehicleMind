# 回放报告技术指标双语展示实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让新生成的回放报告中全部技术指标显示为“英文（中文含义）”，并明确旧报告不会自动更新。

**Architecture:** 只在 `report.js` 的可见文本层集中组合双语标签；`report.html` 保持中文叙述，`summary.json` 与 `trace.json` 保持原协议。使用现有 Node DOM 模拟测试验证实际渲染，再用新的运行标识生成报告；不覆盖旧结果。

**Tech Stack:** 原生 JavaScript、Python 3.13、pytest、Node.js（可用时运行 DOM 测试）。

## Global Constraints

- `summary.json`、`trace.json` 的键、枚举值、工具名、事件类型、文件路径和哈希不改。
- 旧 `runs/drowsy-rest-stop/` 不覆盖、不删除；新结果使用独立运行标识。
- 页面说明、状态、失败原因和证据限制保留中文；技术标签统一为 `English（中文含义）`。
- 未知技术字段显示 `raw_key（未翻译字段）`；`STALE`、`INVALID` 等可见枚举保留原值并附中文释义。
- 录制观测回放不被描述为真实视频推理。

---

### Task 1: 双语报告展示层

**Files:**
- Modify: `apps/vehicle_ai_demo/replay_ui/report.js`
- Modify: `tests/vehicle_ai/replay/test_report_ui.py`
- Test: `tests/vehicle_ai/replay/test_replay_report.py`

**Interfaces:** `report.js` 继续读取 `vehiclemind-data` 中的现有数据；`displayLabel(key)` 返回 `raw_key（中文释义）` 或 `raw_key（未翻译字段）`，`valueText(value)` 不修改原数据。

- [ ] **Step 1: 扩展失败的 DOM 渲染测试。** 在 `test_report_ui.py` 的 Node 模拟数据中加入观测 `confidence`、上下文未知字段、`HIGH_RISK_DETECTED` 事件与 `start_navigation` 工具断言；检查顶部 `Runtime（运行耗时）`、字段 `risk（风险等级）`、状态 `STALE（已过期）`、事件和工具的双语名称，以及未知字段回退。

```javascript
assert.match(allText(roots['headline-metrics']), /Runtime（运行耗时）/);
assert.match(allText(roots['context-grid']), /risk（风险等级）/);
assert.match(allText(roots['context-grid']), /STALE（已过期）/);
assert.match(allText(roots['timeline']), /HIGH_RISK_DETECTED（检测到高风险）/);
assert.match(allText(roots['assertions']), /start_navigation（启动导航）/);
```

- [ ] **Step 2: 运行红灯。** `uv run --group dev pytest -q tests/vehicle_ai/replay/test_report_ui.py`；预期因现有纯中文标签失败。
- [ ] **Step 3: 实现集中映射。** 保留 `labels`、`events`、`tools`、`values` 的中文释义；新增格式化函数，所有技术字段由原始标识与释义拼接，不把双语字符串写入数据对象。

```javascript
function bilingual(raw, translations, fallback = '未翻译字段') {
  return `${raw}（${translations[raw] ?? fallback}）`;
}
function displayLabel(key) { return bilingual(key, labels); }
function eventText(type) { return bilingual(type, events, '未翻译事件'); }
function toolText(name) { return bilingual(name, tools, '未翻译工具'); }
```

顶部四项固定标签、上下文域名、时间线技术字段、事件类型、工具名、断言名称和可见状态枚举均调用同一规则；普通中文句子、用户输入、哈希、路径与 ID 保持原义。避免 `D（前进挡（D））` 等重复释义，挡位释义改为“前进挡”等。
- [ ] **Step 4: 运行绿灯与语法检查。** `uv run --group dev pytest -q tests/vehicle_ai/replay/test_report_ui.py tests/vehicle_ai/replay/test_replay_report.py`、`node --check apps/vehicle_ai_demo/replay_ui/report.js`；预期全部通过。
- [ ] **Step 5: 提交。** `git add apps/vehicle_ai_demo/replay_ui/report.js tests/vehicle_ai/replay/test_report_ui.py tests/vehicle_ai/replay/test_replay_report.py && git commit -m 'Render replay metrics with bilingual labels'`。

### Task 2: 文档、真实报告与交付验收

**Files:**
- Modify: `README.md`
- Modify: `todolist.md`
- Modify: `tests/test_public_docs.py`
- Generate: `runs/drowsy-rest-stop-bilingual/`（本轮验收产物，不提交到 Git）

**Interfaces:** 回放命令保留 `--scenario`、`--output-root`、`--run-id`；本轮验收报告路径为 `runs/drowsy-rest-stop-bilingual/report.html`。README 使用自动生成的运行标识，避免已有本地产物导致复制命令失败。

- [ ] **Step 1: 先写失败的文档测试。** 在 `tests/test_public_docs.py` 检查 README 首屏包含 `assets/demo/cabin_demo.gif` 与 `assets/demo/driving_perception.gif` 的 Markdown/HTML 图片引用、两个文件确实存在，并包含自动生成 `VM_RUN_ID`、传入 `--run-id "$VM_RUN_ID"` 与打开对应报告的命令。运行 `uv run --group dev pytest -q tests/test_public_docs.py`，预期因旧 README 缺少 GIF 和新命令失败。
- [ ] **Step 2: 更新 README 与任务清单。** 在快速启动前直接嵌入仓库现有的舱内、舱外 GIF，附中文说明且注明是感知演示画面，不将其误称为本次回放推理。快速启动段明确“旧报告不会随模板更新；每次运行生成新的 `--run-id`”，提供下列命令，并在 `todolist.md` 记录双语报告和 GIF 恢复情况但不勾选尚无真实指标的作品集目标。

```bash
VM_RUN_ID="drowsy-rest-stop-$(date +%Y%m%d-%H%M%S)-$RANDOM"
uv run --group dev python -m apps.vehicle_ai_demo.replay_demo \
  --scenario assets/scenarios/drowsy_rest_stop.yaml \
  --output-root runs --run-id "$VM_RUN_ID"
open "runs/$VM_RUN_ID/report.html"
```

- [ ] **Step 3: 运行文档绿灯与完整门禁。** `uv run --group dev pytest -q tests/test_public_docs.py`、`uv run --group dev pytest -q`、`uv run --group dev ruff check apps modules scripts tests`、`uv run --group dev ruff format --check apps modules scripts tests`、`uv run --group dev mypy modules/config modules/vehicle_ai/context/enums.py modules/vehicle_ai/integration/cabin_adapter.py scripts/verify_assets.py`、`uv run --group dev python scripts/check_source_size.py`；预期全部成功。
- [ ] **Step 4: 提交代码与文档。** `git add README.md todolist.md tests/test_public_docs.py docs/superpowers/plans/2026-09-23-bilingual-report-metrics.md && git commit -m 'Restore README GIFs and explain bilingual report'`。
- [ ] **Step 5: 生成新报告并核对。** 使用固定的本轮验收标识 `drowsy-rest-stop-bilingual` 单独运行一次，不使用 `--allow-dirty`；确认新 `report.html` 为 `lang="zh-CN"`、新 `report.js` 包含双语映射、`summary.json` 仍保留原始键和工具名，旧 `runs/drowsy-rest-stop/report.js` 仍是历史版本。README 命令使用不同的自动运行标识，供后续重复执行。
- [ ] **Step 6: 审查与 GitHub 同步。** 独立代码审查后推送功能分支、创建并附加 PR，等待 CI 通过，合并后快进同步本地 `main`；向用户提供新报告路径与打开命令，并按 `todolist.md` 告知完成项和下一步。

# VehicleMind 中文介绍与演示实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让项目介绍、回放报告和终端演示的主要可见内容为中文，同时保持机器协议不变。

**Architecture:** 将报告的中文映射与时间线摘要集中在 `report.js`，模板只承载静态中文标签；原始结果数据仍由 `report.py` 按现有 JSON schema 输出。README 缩为中文入口页，详细算法与数据说明拆入中文文档，终端仅翻译交互提示与结果前缀。

**Tech Stack:** Python 3.13、pytest、原生 HTML/CSS/JavaScript、Markdown。

## 全局约束

- `summary.json`、`trace.json` 的键、枚举值、工具名、事件类型、文件路径和哈希不改。
- 页面和说明使用简体中文，专有名词首次出现附中文解释；第三方许可证及引用元数据保持原文。
- 不把录制观测声称为真实视频推理，不编造精度、Agent 成功率或性能数字。
- 所有修改可在无模型、无摄像头、无 API Key 的离线环境验证。

---

### 任务 1：回放报告中文可见层

**文件：**修改 `apps/vehicle_ai_demo/replay_ui/report.html`、`report.js`、`modules/vehicle_ai/replay/report.py`、`assets/scenarios/drowsy_rest_stop.yaml`；测试 `tests/vehicle_ai/replay/test_replay_report.py`、`tests/vehicle_ai/replay/test_replay_runner.py`。

**接口：**生成器仍输出 `report.html`、`summary.json`、`trace.json`；页面从 `vehiclemind-data` 读取原 schema，不向 JSON 注入翻译键。

- [ ] 先添加失败测试：报告 `lang="zh-CN"`、中文标题/限制、中文状态文案；同一次运行的 JSON 键与工具名保持原样。

```python
html = paths.report_html.read_text(encoding="utf-8")
summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
assert '<html lang="zh-CN">' in html
assert "舱内观测" in html and "场景断言" in html
assert "录制的语义观测" in html
assert summary["successful_tools"] == ["search_nearby_rest_area", "start_navigation"]
```
- [ ] 运行 `uv run --group dev pytest -q tests/vehicle_ai/replay/test_replay_report.py`，确认测试因英文可见文案失败。
- [ ] 把模板静态标签、页面 JS 映射与时间线摘要、报告限制、示例场景叙述改为中文；未知字段显示原始键并标注未翻译。

```javascript
const domainLabels = {driver: "驾驶员", road: "道路", vehicle: "车辆"};
const kindLabels = {context_update: "上下文更新", event: "风险事件", tool_result: "工具结果"};
function displayLabel(key, labels) { return labels[key] ?? `未翻译字段：${key}`; }
```

将 HTML 固定标题替换为中文；`report.py` 的 `limitations` 使用“录制的语义观测”“不代表感知精度”“模拟车辆动作”等明确措辞；示例 YAML 的 title、description 和脚本化回复翻译为中文，不改变步骤或工具参数。
- [ ] 运行回放报告与场景测试，生成实际报告并检查中文标题、面板和安全限制，提交该任务。

### 任务 2：中文项目介绍与使用说明

**文件：**修改 `README.md`，创建 `docs/cabin_perception.md`、`docs/road_perception.md`、`docs/project_limits.md`、`tests/test_public_docs.py`；检查 `docs/offline_video_pipeline.md`、`docs/agent_pending_actions.md`。

**接口：**README 提供从仓库根目录可复制的一条离线 Demo 命令和 `open runs/drowsy-rest-stop/report.html`；详细文档从 README 链接可达。

- [ ] 先添加文档检查测试：README 首屏含中文项目定位、录制回放与真实视频区别、启动及打开报告命令、技术贡献/限制；链接目标存在。

```python
readme = (ROOT / "README.md").read_text(encoding="utf-8")
first_screen = readme[:2500]
assert "录制观测" in first_screen and "真实视频" in first_screen
assert "open runs/drowsy-rest-stop/report.html" in readme
for path in ("docs/cabin_perception.md", "docs/road_perception.md", "docs/project_limits.md"):
    assert (ROOT / path).is_file()
```
- [ ] 运行该测试，确认旧 README 的英文主标题/缺失结果查看命令导致失败。
- [ ] 将旧 README 的有效算法、数据来源、许可证和运行信息分配到三个中文专题文档，首页改为简洁中文说明；删除过时重复段落，不删除法律/引用原文。

README 固定为以下中文顺序：定位与项目边界、两种离线运行方式、演示结果怎么看、系统模块和个人实现、复现与验证、限制与第三方许可、专题文档链接。`docs/cabin_perception.md` 保留关键点/PERCLOS/哈欠/状态判定说明，`docs/road_perception.md` 保留目标/车道/可行驶区与推理条件，`docs/project_limits.md` 集中记录数据来源、模型许可边界、当前未完成项。三个文件都须是完整中文叙述，不复制旧英文标题。
- [ ] 运行文档检查和完整测试，提交该任务。

### 任务 3：中文终端提示与收尾验收

**文件：**修改 `apps/vehicle_ai_demo/replay_demo.py`、`integrated_demo.py`、`video_pipelines.py`；创建 `tests/vehicle_ai/replay/test_replay_cli.py`，修改 `tests/vehicle_ai/test_integrated_video_options.py`。

**接口：**CLI 参数、`context/events/health/quit` 命令及进程退出码保持兼容，仅面向用户的提示中文化。

- [ ] 先添加失败测试：回放成功/失败的中文状态与报告路径前缀，以及交互菜单的中文解释；断言原 CLI 参数仍可解析。

```python
status = "通过" if result.passed else "失败"
assert "报告页面：" in captured.out
assert "结果摘要：" in captured.out
assert "context  - 查看统一上下文" in menu_output
assert parse_args(["--perception-only"]).perception_only is True
```
- [ ] 运行聚焦测试，确认英文输出导致失败。
- [ ] 改写终端提示、感知摘要、帮助文案；在线模型提示词要求中文回复，但不承诺外部模型始终遵守。

```python
print(f"{status}：{scenario.scenario_id}")
print(f"报告页面：{paths.report_html.resolve()}")
print(f"结果摘要：{paths.summary_json.resolve()}")
```

交互命令只改说明文本，例如 `context  - 查看统一上下文`；保留 `context`、`health` 等输入值和进程退出码。感知卡片把 `presence/state/vehicles/lane` 等前缀变成中文，不改变 `PerceptionCard` 字段。
- [ ] 运行完整 pytest、Ruff、mypy、文件长度检查；独立代码审查后推送 PR、等待 CI、合并并同步本地 `main`。

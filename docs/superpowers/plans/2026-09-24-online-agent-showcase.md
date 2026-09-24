# 在线 Agent 决策展示 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 单次在线 Agent 运行自动生成可直接打开的中文决策展示页。

**Architecture:** 从已有 EvaluationCase/TrialResult 构建只读展示投影；独立 HTML 生成器输出本地页面；`write_report` 连接入口，原始 JSON 不变。

**Tech Stack:** Python 3.13、标准库 HTML/JSON、pytest、Ruff。

## Global Constraints

- 录制观测与真实视频严格区分。
- 只显示实际模型输入与输出，不推断未发生的建议或动作。
- 不改变 Agent 或工具执行语义。

---

### Task 1: 展示数据投影与 HTML

**Files:** Create `modules/vehicle_ai/evaluation/showcase.py`; Test `tests/vehicle_ai/evaluation/test_showcase.py`.

**Interface:** `render_showcase(case: EvaluationCase, trial: TrialResult, grade: dict) -> str` 输出完整 HTML。

- [ ] 写测试：断言真实首轮上下文、闭眼/哈欠/道路字段、回复、工具、导航状态和来源声明出现；恶意文本被转义；缺失观测显示未提供。
- [ ] 运行 `uv run --group dev pytest tests/vehicle_ai/evaluation/test_showcase.py -q`，确认因缺少接口失败。
- [ ] 实现只读投影和可访问的卡片式 HTML，原始证据放折叠区。
- [ ] 重跑定向测试并通过 Ruff。

### Task 2: 接入单次报告与使用说明

**Files:** Modify `modules/vehicle_ai/evaluation/report.py`, `modules/vehicle_ai/evaluation/cli.py`, `README.md`; Test `tests/vehicle_ai/evaluation/test_evaluation.py`.

**Interface:** 每个单次运行目录新增 `report.html`，终端打印路径。

- [ ] 先扩展测试断言 HTML 文件存在并含正确标识，确认失败。
- [ ] 接入 `render_showcase`，保留 `trial.json` 和 `report.md`。
- [ ] 更新 README 的在线 Agent 命令和来源说明。
- [ ] 运行在线评测测试、全套离线测试、Ruff 与格式检查。

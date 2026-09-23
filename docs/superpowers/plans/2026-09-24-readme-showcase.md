# VehicleMind README Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 GitHub 首页改为中文、视觉优先且证据可追溯的作品集展示。

**Architecture:** 不改业务代码。用现有离线回放生成真实报告截图；根目录 README 使用 GitHub Markdown、Mermaid 和仓库图片展示完整链路。用文档测试与真实命令核查资产和措辞。

**Tech Stack:** Markdown、Mermaid、Python 3.13、pytest、Chrome headless、现有 replay CLI。

## Global Constraints

- 参考 `docs/superpowers/specs/2026-09-24-readme-showcase-design.md`，只借鉴 PaperBase 的排版，不复制文案或结果。
- 所有展示文案中文；代码协议、命令和技术名保留原样。
- 不将录制观测写成真实视频推理，不将候选 pilot 写成正式成功率，不新增 Web 应用。
- 用户已要求在当前 `codex/agent-eval-design` 分支工作，并在完成一个节点后停下汇报。

---

### Task 1: 可信的协同演示截图

**Files:**
- Create: `assets/demo/vehiclemind_report.png`
- Inspect: `apps/vehicle_ai_demo/replay_demo.py`, `assets/scenarios/drowsy_rest_stop.yaml`

**Interfaces:**
- Consumes: replay CLI 生成的 `runs/<run-id>/report.html`。
- Produces: README 可直接引用的 `assets/demo/vehiclemind_report.png`。

- [x] **Step 1: 检查运行环境**：确认 `.venv/bin/python` 与 `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` 存在，并运行 `git status --short`，预期无未授权改动。
- [x] **Step 2: 从干净提交运行回放**：执行 `.venv/bin/python -m apps.vehicle_ai_demo.replay_demo --scenario assets/scenarios/drowsy_rest_stop.yaml --output-root runs --run-id readme-showcase-<unique>`；预期产生 `report.html` 与通过的断言。
- [x] **Step 3: 捕获真实页面**：使用 Chrome headless 截取 `assets/demo/vehiclemind_report.png`（1440×2200，可见最终上下文）与 `assets/demo/vehiclemind_report_full.png`（1440×4200，可点击查看时间线、确认及断言），检查图片尺寸与可见内容；若环境不能生成有效截图，则不提交图片，并在 README 中只用已有两个 GIF。

### Task 2: README 首页结构与验证

**Files:**
- Modify: `README.md`
- Modify: `todolist.md`
- Create: `tests/test_readme_showcase.py`

**Interfaces:**
- Consumes: 规格、现有两个 GIF、Task 1 的截图（如有效）、技术报告和文档链接。
- Produces: GitHub 首页和可重复执行的资源/措辞检查。

- [x] **Step 1: 先写失败的文档测试**：在 `tests/test_readme_showcase.py` 检查 README 含 `## ✨ 项目简介`、`## 🌟 核心能力`、`## 🧠 系统架构`、`## 📊 结果与证据`、`## 🚀 快速开始`，两个 GIF 路径、Mermaid、真实 API pilot 的“候选”标识，以及所有相对图片路径在仓库存在。
- [x] **Step 2: 运行红灯**：执行 `.venv/bin/python -m pytest -q tests/test_readme_showcase.py`；预期因当前 README 缺少上述结构而失败。
- [x] **Step 3: 改写 README**：按规格排列标题、徽章、目录、三类演示（截图有效时）、能力表、单张 Mermaid 架构、技术栈/个人工作、候选 pilot 证据、快速开始、项目结构、设计决策和限制。所有数值引用已有报告，缺失指标不填。
- [x] **Step 4: 运行绿灯和回放检查**：执行 `.venv/bin/python -m pytest -q tests/test_readme_showcase.py`、全量 pytest、Ruff、`git diff --check`；核对 README 中链接指向的真实文件。
- [x] **Step 5: 更新清单并提交**：只勾选本次已完成的展示子项，保留“真实指标齐全的最终首屏”未完成；提交并推送当前分支，确认本地与远端 commit 一致。

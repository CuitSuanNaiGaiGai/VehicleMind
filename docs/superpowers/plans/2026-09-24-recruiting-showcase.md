# VehicleMind 招聘展示闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让招聘方从 GitHub 首页进入后，能够理解业务场景、看到可复现 Demo、辨认项目内技术贡献，并核验 Agent 指标与局限。

**Architecture:** 以现有 `README.md` 为导航首页，以离线 `replay_demo` 作为无密钥可运行主案例；中文技术说明、在线 Agent 内部评测报告和一页面试卡承接深读。优先修改文档与文档契约测试，不新建 Web 前端或伪造感知指标。

**Tech Stack:** Markdown、Mermaid、Python 3.13、pytest、Ruff、现有 YAML 场景与 HTML 回放报告。

## Global Constraints

- 真实视频感知与录制语义观测回放必须分别标注。README 截图来自后者，两个 GIF 是独立感知演示素材；不得声称三者为同一次推理。
- 40 条冻结场景 × 每模型 3 次是 AI 自审内部评测，不是独立人工 Golden Set；80 条变体只用于开发回归。仅引用已公开、可核验的结果。
- 导航仅为模拟工具；无密钥主 Demo 不依赖摄像头、模型权重和在线 API。在线入口需说明凭证与费用。
- 完成一项即核对 `todolist.md` 对应验收条件、运行相关验证、提交并推送；仅在真实满足时勾选。沿用当前功能分支，不创建额外 worktree。
- 不延伸到小样本感知评测、性能测试、Release 或视频制作；这些仍列为后续任务。

---

### Task 1: 首页业务故事与路径辨识

**Files:** Modify `README.md`, `tests/test_readme_showcase.py`, `todolist.md`.

**Interfaces:** README 首屏链接到 `assets/scenarios/drowsy_rest_stop.yaml`、演示截图、架构与证据；图片仍引用 `assets/demo/` 的现有资产。

- [ ] 先在 `tests/test_readme_showcase.py` 增加断言：首屏包含具体疲劳驾驶场景、两种输入路径及“模拟导航”边界，截图/GIF 的不同来源有明确文字。执行 `uv run --group dev python -m pytest -q tests/test_readme_showcase.py`，确认因缺少新内容而失败。
- [ ] 精简并调整 `README.md` 首屏顺序：具体场景 → 录制观测回放与真实视频感知的分叉 → 风险事件/Agent/确认/模拟导航 → 可点击演示、架构和结果。不要增加未经核验的数值。
- [ ] 重跑目标测试、`git diff --check`；人工检查 GitHub Markdown 锚点和图片路径；满足清单的首项验收后更新 `todolist.md`，提交并推送。

### Task 2: 从零运行的中文演示说明

**Files:** Modify `README.md`, `docs/offline_video_pipeline.md`, `tests/test_readme_commands.py`, `todolist.md`.

**Interfaces:** 无密钥入口 `python -m apps.vehicle_ai_demo.replay_demo`；可选真实视频入口 `python -m apps.vehicle_ai_demo.integrated_demo`；可选在线 Agent 入口以仓库现有 CLI 为准，不新增未实现命令。

- [ ] 先扩充 `tests/test_readme_commands.py`：检查离线命令、报告路径、示例场景、预期可见状态，以及在线路径的 `.env`/费用提示。运行目标测试，确认新增断言红灯。
- [ ] 用临时唯一 `--run-id` 执行 README 无密钥命令，检查 `report.html`、`summary.json`、`trace.json`、`resolved_config.yaml` 和断言结果；记录确切命令与输出，不把本地运行目录提交 Git。
- [ ] 把实际验证过的安装、打开报告、预期画面、常见失败处理写入 README。核对 `docs/offline_video_pipeline.md` 的可选真实视频命令与代码参数；在线演示只链接确实可运行且明确计费的现有指南。
- [ ] 运行目标测试、离线 smoke test 与 `git diff --check`；按验收更新 `todolist.md`，提交并推送。

### Task 3: 业务模块—技术栈—个人贡献映射

**Files:** Modify `README.md`, `tests/test_readme_showcase.py`, `todolist.md`; inspect `docs/cabin_perception.md`, `docs/road_perception.md`, `docs/agent_pending_actions.md` and relevant `modules/` implementations.

**Interfaces:** README 表格逐行提供“业务问题 / 输入输出 / 核心技术 / 项目内工作 / 源码或中文说明”五个可核查维度。

- [ ] 先为表格覆盖的舱内、舱外、上下文/事件、Agent、工具门禁、回放/测试添加文档契约断言；运行目标测试确认红灯。
- [ ] 核查对应源码后改写技术栈表。明确 MediaPipe、YOLOPv2、ONNX Runtime 与 Qwen/GLM 是第三方能力，项目内工作是管线、适配、状态处理、编排、门禁和验证；每行增加可点击代码或技术说明链接。
- [ ] 运行目标测试、链接存在性检查和 `git diff --check`；满足验收后更新 `todolist.md`，提交并推送。

### Task 4: 成功与失败的证据闭环

**Files:** Modify `README.md`, `tests/test_readme_showcase.py`, `todolist.md`; inspect `assets/scenarios/`, `docs/reports/2026-09-24-online-agent-internal-evaluation.md`, `docs/project_limits.md`.

**Interfaces:** 首页至少一条成功回放链路、一条仓库内可运行的失败/拒绝链路；指标链接到中文报告，案例链接到场景或测试。

- [ ] 选定现有失败或拒绝场景，先执行回放并核对其断言与报告；若没有合适场景，先写一个失败的 smoke test，再增加最小 YAML 场景，不造假截图。
- [ ] 在 `tests/test_readme_showcase.py` 增加证据契约：成功/失败入口都可达、Agent 数值具有分子/分母和 AI 审核边界、未出现未完成的感知精度与性能成绩；执行红灯。
- [ ] README 用并排或连续短案例展示期望状态、实际结果及对应证据；公开文案不得含密钥、本机绝对路径或个人轨迹。运行目标测试、场景 smoke、`git diff --check`；按验收更新清单并提交推送。

### Task 5: 一页面试讲述卡与总体验收

**Files:** Create `docs/interview_story.md`, `tests/test_interview_story.py`; modify `README.md`, `todolist.md`.

**Interfaces:** README 链接中文讲述卡；卡片的数字与报告、场景、可复现命令一一对应。

- [ ] 先写 `tests/test_interview_story.py`：检查卡片包含业务场景、架构取舍、个人模块、一次真实失败修复、可信数字、局限和 60–90 秒提纲，所有相对链接均存在；运行红灯。
- [ ] 从现有报告和提交记录选一项可核验的 Agent 失败定位/修复，写入一页卡片；不得把模拟车机称为真实控制，不把内部 AI 自审称为人工金标。README 增加清晰入口。
- [ ] 运行 `uv run --group dev python -m pytest -m 'not hardware and not online' -q`、Ruff check/format、`python scripts/check_source_size.py`、`git diff --check`；从 README 逐项人工核对五条展示验收。更新 `todolist.md` 后提交推送，创建 PR 并等待必需 CI，再考虑合并 main。

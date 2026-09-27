# Agent-Focused Project Positioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the VehicleMind README and interview story so they present a LangGraph-based multimodal cockpit Agent system and make its Agent engineering value clear.

**Architecture:** Keep the change documentation-only. Put the user problem, Agent lifecycle, permission boundary, recovery, structured trace, and evaluation near the top of the README; present perception, RAG, and trip memory as supporting inputs. Make the interview story a coherent short pitch backed by existing reports.

**Tech Stack:** Markdown, LangGraph, Git, existing offline replay and Agent evaluation reports.

## Global Constraints

- README 标题清楚写出“基于 LangGraph 的多模态智能座舱 Agent 系统”。
- 首页首屏能直接回答项目解决的问题、Agent 的关键设计和可体验入口。
- 页面主线以 Agent Runtime、执行授权、恢复、Observability / Evaluation 为中心。
- 感知、RAG、记忆作为系统输入或支持能力出现，不与 Agent 主线争夺篇幅。
- 面试讲述卡形成连贯的 60–90 秒叙述，含一项工程故障复盘和可核验结果。
- 所有现存评测失败记录和原始报告链接保留；不产生未经证据支持的数字或能力声明。
- 现有快速开始命令与证据链接继续有效。

---

## File Structure

- `README.md` — GitHub homepage, demo entry points, Agent architecture, compact evidence index, and links to subsystem docs.
- `docs/interview_story.md` — interview narrative, ownership, one failure investigation, evidence, and a 60–90 second pitch.
- `docs/superpowers/specs/2026-09-27-agent-project-positioning-design.md` — approved narrative design; reference only, do not revise during implementation.

## Task 1: Make the README Agent-Centered

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: existing scenario YAML files, LangGraph design guide, evaluation reports, and current quick-start commands.
- Produces: a homepage whose first screen names the system, frames the Agent problem, and points to a runnable demo.

- [ ] **Step 1: Replace the title, subtitle, and opening overview** with “基于 LangGraph 的多模态智能座舱 Agent 系统” and a concise account of how cabin/road observations, vehicle state, and user requests enter the semantic context and Agent workflow.
- [ ] **Step 2: Reframe the lead demo** around one lifecycle: fatigue context → risk event → service-area search → PendingAction → user approval → simulated navigation result. Keep links to the existing replay scenario and screenshots.
- [ ] **Step 3: Replace the capability table and architecture diagram** with an Agent control-flow view: context/event input → LangGraph StateGraph → VehicleAgent → ToolRegistry policy → approval interrupt/resume → action result, with structured trace and evaluation attached. Show perception, LightRAG, and SQLite trip memory as support modules.
- [ ] **Step 4: Add a compact Agent engineering highlights block** covering stateful orchestration, execution-time authorization, bounded recovery with renewed approval, and correlated traces/evaluation. Link each point to its existing implementation or guide.
- [ ] **Step 5: Run a content diff review** with `git diff -- README.md`; confirm the complete title is present, the three replay commands remain unchanged, and the diagram reflects resume as workflow continuation while ToolRegistry remains the execution boundary.

## Task 2: Condense Homepage Evidence Without Losing Failure Records

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: current A5/A6/RAG evaluation reports and preserved failure cases.
- Produces: concise top-level evidence plus links to full methodology, reports, and failures.

- [ ] **Step 1: Reduce the visible evidence section** to representative deterministic workflow evidence, the existing Qwen/GLM A6 evaluation summary, and the structured trace implementation/tests.
- [ ] **Step 2: Preserve evaluation failure access** by keeping the R03 case and complete internal evaluation report linked; keep the existing detailed reports in the repository unchanged.
- [ ] **Step 3: Keep supporting modules discoverable** through short links to the perception, LightRAG, trip-memory, and plan-recovery guides rather than repeating their full technical breakdowns.
- [ ] **Step 4: Check every changed relative Markdown link** against its target file and keep the existing replay quick-start commands intact.

## Task 3: Rewrite the Interview Story

**Files:**
- Modify: `docs/interview_story.md`

**Interfaces:**
- Consumes: the same implementation and result evidence linked from `README.md`.
- Produces: an interview-ready explanation of the project problem, Agent design, ownership, failure repair, and evidence.

- [ ] **Step 1: Open with the exact project positioning** and a one-paragraph problem/solution narrative centered on stateful Agent execution under changing cabin context.
- [ ] **Step 2: Explain the four connected responsibilities**: StateGraph workflow state, bounded VehicleAgent model/tool loop, ToolRegistry/PendingAction authorization, and trace/evaluation.
- [ ] **Step 3: Keep the M01 failure investigation** and describe the correction from treating `CONFIRMATION_REQUIRED` as tool failure to preserving the pending-approval state.
- [ ] **Step 4: Retain a small set of verified results** with links to the A5/A6 source reports and write one 60–90 second pitch without repetitive disclaimers.
- [ ] **Step 5: Run `git diff -- docs/interview_story.md`** and check that each numeric claim still matches its linked source report.

## Task 4: Remove Repeated Defensive Wording from the Homepage

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the approved Agent-focused homepage and its existing demo commands.
- Produces: concise, factual run-mode descriptions that preserve operating instructions without repeated negation or caveat blocks.

- [ ] **Step 1: Rewrite the input-path and demo captions** as direct descriptions of local-video perception and recorded-observation replay; preserve what each path runs and where it enters the Agent workflow.
- [ ] **Step 2: State runtime behavior positively and precisely**: name the process-local `InMemorySaver` and optional trace export behavior without contrasting it with a production system.
- [ ] **Step 3: Tighten explanatory prose around replay and optional modules** while preserving every current run command, target scenario, and guide link. Move detailed perception-audit procedures behind a disclosure block if needed to keep the Agent homepage concise.
- [ ] **Step 4: Replace the limitations-heavy ending** with a short “运行范围与后续方向” section describing the three project run modes, simulated tool state, and existing roadmap link in plain factual language.
- [ ] **Step 5: Run `git diff -- README.md`** and confirm only explanatory prose/headings changed; all shell commands, scenario paths, report links, and assets remain present.

## Task 5: Verify and Commit the Documentation Update

**Files:**
- Review: `README.md`
- Review: `docs/interview_story.md`

**Interfaces:**
- Consumes: completed homepage and interview-story edits.
- Produces: reviewed documentation on the feature branch with no runtime changes.

- [ ] **Step 1: Run `git diff --check`**; expected result is exit code 0 with no whitespace errors.
- [ ] **Step 2: Check changed relative links**; expected result is that every local Markdown/image/scenario target exists.
- [ ] **Step 3: Review preservation requirements** with `git diff --name-only` and confirm only the two approved content files changed after the plan/spec commits; verify all cited historical evaluation reports remain present and untouched.
- [ ] **Step 4: Run the requested repository verification**: `uv sync --frozen --group dev`, `uv run ruff check`, `uv run ruff format --check`, and `uv run pytest -q`; record actual command results without converting failures into success claims.
- [ ] **Step 5: Commit the documentation change** with `git add README.md docs/interview_story.md && git commit -m "docs: focus project story on LangGraph Agent system"`.

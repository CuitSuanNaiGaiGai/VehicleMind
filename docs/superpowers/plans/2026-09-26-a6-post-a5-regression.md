# A6 A5 后回归证据实施计划

## 目标

依照 [设计](../specs/2026-09-26-a6-post-a5-regression-design.md)，复用冻结 Agent 场景，生成可复核的同口径前后报告，并更新公开入口。在线数据仅限合成场景与合成回复。

## 技术栈

Python 3.13、现有 `batch_cli` / `batch_judge_cli`、SHA-256 冻结 YAML、pytest、JSON/Markdown/HTML、Qwen/GLM API；不增加依赖，不上传 `runs/` 原始记录。

## 执行步骤

### 1. 评测 provenance

- Files: `modules/vehicle_ai/evaluation/batch.py`, `modules/vehicle_ai/evaluation/batch_cli.py`, `tests/vehicle_ai/evaluation/test_batch.py`, `tests/vehicle_ai/evaluation/test_batch_cli.py`
- 先写失败测试：run metadata 收到并持久化；不传时向后兼容；密钥类字段不允许进入 provenance。
- 最小实现：保存 `source_revision`、manifest/rubric digest、模型/温度/timeout/Agent budgets 及 case 集大小/分割；run 与现有逐 trial trace 关联。
- 运行专项测试、ruff 与 CLI 测试。

### 2. 报告汇总器

- Files: `modules/vehicle_ai/evaluation/regression_report.py`, `modules/vehicle_ai/evaluation/regression_report_cli.py`, `tests/vehicle_ai/evaluation/test_regression_report.py`
- 先写失败测试：trial 缺失、case 哈希不匹配、审核源不匹配时拒绝报告；全量分母和 p50/p95 可重算；N/A usage 不当成 0；异常与未审核保留。
- 生成 summary JSON、中文 Markdown/HTML；机械成绩与当前协议语义成绩分开，案例为单位的 N 与重复数显式并列。
- HTML 只展示脱敏聚合结果、运行 ID 和短哈希，不渲染 prompt、全量 trace 或本机路径；A3/A4/A5 分栏引用，不做复合得分。
- 运行专项测试及 HTML escaping 测试。

### 3. 同协议前后数据

- Files: `runs/agent_eval/`（Git 忽略目录）、必要时 `scripts/` 下的可复现命令；不跟踪运行原始产物。
- 固定基线：`batch-20260924T032532645475Z-qwen` 与 `batch-20260924T032544374590Z-glm`；先校验 40 case/rubric 哈希与全部 120 条 trace。
- 使用当前审核协议 v4 分别复核旧基线与 A5 后新 run，保留每批自己的 judge progress 和 source hash；旧原始目录不得被覆盖。
- 启动 Qwen/GLM 各 40 × 3 全量新 run；记录运行 commit/hash、模型配置、request/token/latency 和每条机械结果。
- 对比任何模型语义成功变化前，确认同一 rubric、同一 judge protocol 和 complete 120 decisions；AI review 仅标为内部辅助审核。
- 历史 trace 缺少的执行预算保留为未知，不用当前默认值补写；已知预算必须一致。预算不全时只并列展示描述性结果，提升差值为 N/A，不归因于代码优化。
- 审核响应先持久化再解析；格式错误保留原文摘要与 usage，按失败计入而不自动重试挑选结果。审核支持从已保存 case 续跑；已发生但无法追回 usage 的调用显式记为未核算，费用为 N/A。
- Agent runner 保留逐 trial API 失败并继续；进程中断产生的不完整批次保留并排除汇总，不宣称支持逐 trial 断点续跑。审核进程中断可恢复已保存进度。

### 4. 费用口径与报告内容

- Files: `docs/reports/2026-09-26-a6-online-regression.md`, `docs/reports/2026-09-26-a6-post-a5-regression.html`（或生成器输出到忽略目录，再复制仅汇总 html 到报告目录）, `docs/reports/2026-09-26-a6-showcase-and-regression.md`
- 查证两家 configured endpoint 对应官方费率；仅当价格、币种、地域、上下文计费档可核实时估算费用并注明非账单金额，否则为 N/A。
- 展示 Task Success、mechanical tool/argument/state match、错误数、审核待处理数、p50/p95、请求/token 和 A3/A4/A5 分项证据；保留成功/失败例链接到可公开固定案例/rubric，不公开模型私有请求全文。
- 记录约束：合成小样本、非人工真值、非实车安全、review 同源偏差、感知准确率未评估。

### 5. 文档闭环和交付

- Files: `README.md`, `docs/interview_story.md`, `todolist.md`, A6 reports/spec/plan
- 给出本地运行与产物定位命令；更新 TODO 中已由新证据满足的项，保留感知标签、路类映射语义人工复核和 Release 后置项。
- 运行专项与全量 pytest、Ruff lint/format、CI 指定 mypy、源码大小策略、报告/HTML 安全回归、`git diff --check`。
- 提交代码和公开汇总文档，不提交 `.env`、`runs/`、媒体或 raw judge/request trace；推送 PR、通过 CI 后合并 main、同步本地 main。

## 停止条件

- 冻结用例含真实个人数据/本机媒体、case/hash 不一致、API 凭据缺失/权限错误、评分对照无法做到同口径，或 PR 保护/CI 无法通过。
- 停止时保留已完成安全部分，记录哪些未完成和证据；不以猜测补齐结果。

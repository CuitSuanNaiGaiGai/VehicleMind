# A6 贯穿式评测与招聘展示实施计划

**目标：**把 A5 计划过程加入现有双语在线 HTML；复核旧评测集完整性；更新 README、面试讲述卡与总 TODO，提供可分辨层级、命令、指标分母和局限的展示闭环。

**技术栈：**沿用 Python、pytest、现有冻结 YAML loader / batch runner / grader、在线 Qwen/GLM 客户端、JSON trace、中文 HTML/CSS、GitHub Actions。不新增依赖或统一聚合不同实验口径的总分。

## Task 1：在线报告展示任务计划

- Files: `modules/vehicle_ai/evaluation/task_display.py`, `tests/vehicle_ai/evaluation/test_a1_trace.py`
- [x] 先写失败测试：有 plan trace 时显示步骤/预算/候选/恢复，旧 trace 缺 plan 时兼容。
- [x] 实现全部来自 trace 的中文计划摘要；保留 GIF、关键事实、实际模型输入和折叠原始 JSON。
- [x] 运行展示相关测试并验证 HTML 对用户内容做转义。

## Task 2：在线内部回归与 A5 在线展示试跑

- Files: `runs/agent_eval/`（本地忽略目录）、`docs/reports/2026-09-26-a6-online-regression.md`
- [x] 冻结评分口径并检查 40 条 manifest、case/rubric 哈希；将旧 heldout 明确记为已审阅的历史集。
- [ ] 使用现有 batch runner 对原 40 条场景做 Qwen / GLM 各 3 次回归，保存全部原始 trace、usage 与 latency；复核与运行提交对应。
- [ ] 对影响最大的旧失败场景按同一评分口径运行修正后 trial，单独标记为定向回归而不混入原分母。
- [ ] 对 `SHOWCASE01` 用 Qwen / GLM 各 3 次运行，保留 needs_review 状态及所有失败；报告请求、token、延迟与可核对工具动作。
- [ ] 按实际 provider/model 官方价格及可用 usage 字段计算成本；缺字段时显示未提供，不估造价格或 token。

## Task 3：招聘首页与讲述闭环

- Files: `README.md`, `docs/interview_story.md`, `docs/reports/2026-09-26-a6-online-regression.md`, `todolist.md`
- [x] 列出完整离线 Demo、Agent 恢复评测、在线模型单例/重复评测、感知零标注核验的启动命令与结果位置。
- [x] 指标注明分子/分母、数据/审核类型、硬件或模拟边界；A1–A5 不合并成一个综合成功率。
- [x] 对旧 TODO 以现有证据复核；无标签精度、数据授权、未做的视频和未实现目标保留为后置。

## Task 4：最终验证与集成

- [ ] 全量离线 pytest、ruff lint/format、CI 指定 mypy、源码行数和 `git diff --check`。
- [x] 从 README 命令重新生成三个离线主案例；README 图片本地目标存在，现有报告回归测试覆盖舱内/舱外 GIF。
- [ ] 每个任务完成后形成独立 commit，推送分支、创建 PR，通过 CI 后合并 `main` 并核对远端提交。

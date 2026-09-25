# 行程事件记忆演示

行程事件记忆把当前行程中已发生的风险、主动提醒、用户请求、确认/取消和车机动作结果写入本地 SQLite。Agent 仅在用户询问历史时调用只读 `query_trip_events`；检索到的历史不会更新 ContextManager，也不能替代当前感知。

## 启动

在仓库根目录配置好 `.env` 中的 Qwen 或 GLM API 后，运行离线视频协同演示：

```bash
uv run python -m apps.vehicle_ai_demo.integrated_demo \
  --cabin-video data/perception/cabin/example.mp4 \
  --road-video data/perception/road/example.mp4 \
  --trip-id demo-trip-001
```

请把示例视频路径替换成本机已有视频。默认数据库位于 `runs/state/trip_events.sqlite3`，被 Git 忽略，不会上传视频或历史记录。省略 `--trip-id` 会创建新行程并在终端打印 ID；需要跨进程继续查询时，重新启动时传入同一个 ID，或设置 `VEHICLEMIND_TRIP_ID`。

## 验证

在提示符输入以下问题，Agent 会按需检索本次行程：

- “本次行程提醒过几次？”
- “刚才取消的是哪个操作？”
- “最近一次动作的执行结果是什么？”

输入 `history` 可查看最近 20 条带时间戳的结构化事件。输入 `clear-history` 并再次输入当前行程 ID，可清空本次行程记录；默认最多查询 50 条，启动时清理超过 30 天的旧事件。

再次触发高风险提醒时，建议 Agent 会参考最近 3 条相关提醒/动作选择，避免忽略用户刚才取消或确认的操作；历史只影响建议表达，不覆盖当前感知事实。

该功能量化关注 Event Retrieval Accuracy（历史事件检索正确率）与 Temporal Confusion（时间混淆次数）。查询结果准确性以冻结结构化事件序列核对；当前状态隔离测试会检查旧高风险记录没有被注入当轮当前车况。它不是独立人工标注的自然语言问答成功率。

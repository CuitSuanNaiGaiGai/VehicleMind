# A4 行程内事件记忆验收

## 范围

本项为 VehicleMind Agent 加入可选 SQLite 行程记忆：持久化语义风险、主动提醒、用户请求、动作确认/取消与工具结果；`query_trip_events` 将行程 ID 固定在运行时，只提供最多 50 条的只读历史检索。保留 30 天，启动时清理过期事件。当前 Driver / Road / Vehicle 状态仍来自 ContextManager，不由历史记忆回填。

高风险重复提醒会额外带入最近 3 条历史提醒/确认/取消/动作结果作为措辞上下文；该片段经过字段筛选与长度截断，并明确标注为历史，不参与风险值计算，也不会降低新的安全提示要求。

## 固定事件序列核验

事件序列包含 4 条当前行程记录（风险、两次提醒、一次取消）以及 1 条其他行程提醒。冻结的查询检查了提醒计数、取消记录、时间窗筛选、完整行程的时间排序与事件集合。

| 指标 | 结果 | 口径 |
|---|---:|---|
| Event Retrieval Accuracy（历史事件检索正确率） | 8/8 字段匹配（4/4 查询） | 每个查询分别核对总数和事件 ID 集；测试序列是结构化合成样本，不是自然语言金标 |
| Temporal Confusion（时间混淆次数） | 0/1 注入检查 | 旧 HIGH 风险记录没有进入“当前驾驶状态”模型输入；历史工具显式标记 `trip_history_not_current_state` |
| Trip Isolation（行程隔离） | 通过 | 其他行程事件不会出现在当前行程查询中 |
| Deduplication（事件去重） | 通过 | 重复事件 ID 不会增加计数 |
| Persistence（持久化） | 通过 | 关闭并重建 Store 后仍可读取；清理和按行程清空可验证 |

## 复现

```bash
uv run pytest tests/vehicle_ai/test_trip_event_store.py \
  tests/vehicle_ai/test_trip_memory_tool.py \
  tests/vehicle_ai/test_trip_memory_acceptance.py \
  tests/vehicle_ai/test_trip_memory_display.py -q
```

真实视频交互入口与行程 ID 复用、历史查看和清空方法见[行程事件记忆演示指南](../guide/trip-memory-demo.md)。这里的评测验证事件存储和结构化检索接口，不代表在线 LLM 的自然语言答案成功率或真实车辆安全保证。

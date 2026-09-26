# LightRAG 与 LLM Agent：实际技术链路与三个核心指标

核查日期：2026-09-26。对应生产代码 `940dccc`。本文以项目源码、已安装的 LightRAG 1.5.7 源码、实际索引和服务 `/health` 为依据，不把库支持的可选功能写成已经启用。

## 1. 为什么选择这三个指标

没有适用于所有 Agent 的唯一“三大指标”。本项目的展示目标是验证三个不同问题：证据能否找到、引用是否成立、任务是否完成。因此选择：

| 主指标 | 计算口径 | 回答的问题 |
|---|---|---|
| Evidence Hit@5（前五条证据命中率） | 可回答题中，前五条 chunk 的来源 ID 与预期来源集合有交集的题数 / 全部可回答题数；检索失败仍留在分母 | LightRAG 能否找回预期证据？ |
| Citation Support Rate（引用支持率） | 被引用片段支持的带引用事实子句数 / 全部已审带引用事实子句数 | Agent 是否正确使用了它实际取得的证据？ |
| Task Success Rate（端到端任务成功率） | 工具/参数/最终状态检查与回答语义审核同时通过的 trial 数 / 所有预定 trial 数 | Agent 是否真正完成业务目标，而不只是生成合法工具调用？ |

选择依据：

- [Ragas Context Recall](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/context_recall/) 提供按证据 ID 计算召回的方法。本项目当前 20 条可回答题各指定一个预期来源，因此来源级 Recall@5 与 Hit@5 数值相同；但代码只判断“至少一个命中”，不是一般多证据召回率。历史 JSON 的 `recall_at_5` 保留兼容，本文使用精确名称 Hit@5。
- [ALCE（EMNLP 2023）](https://aclanthology.org/2023.emnlp-main.398/) 强调引用质量需要单独评估，不能由答案流畅度代替。本项目的引用支持率是自定义 AI 辅助逐子句核验，不是运行 ALCE 官方评测器。
- [τ-bench](https://arxiv.org/abs/2406.12045) 以任务结束时的状态是否满足目标来检查 Agent。本项目借鉴结果导向，但采用自己的模拟车机断言和冻结语义 rubric，不宣称取得 τ-bench 成绩。

**引用支持率不是整段回答 Faithfulness（忠实度）。** [Ragas Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) 检查回答中的全部事实主张；现有审查器仅检查带 `[Kxxx]` 的事实子句，不能发现所有未引用的幻觉。报告必须同时披露实际检索覆盖、可审题数、无引用题数和审核错误，不能把无引用/审核失败当作 100% 支持。

本轮不把 MRR/nDCG、延迟、tokens、弃答等都塞进三个主指标。当前只有单一预期来源标签，没有完整的分级相关性标注；nDCG 暂不作为主指标。延迟/用量是成本诊断，弃答与范围隔离是必要旁证，另行列出。

## 2. 语料是什么，如何入库

[`source_catalog.yaml`](../../config/knowledge/source_catalog.yaml) 管理 20 条 Markdown 知识卡片，记录来源 URI、章节、profile、版本与 SHA-256。它们是简短知识片段，不是一套完整汽车手册库。

- `vehicle_common`：K001–K010，10 条通用车辆/驾驶知识。
- `vehiclemind_demo`：K001–K020，20 条通用与本项目实现知识。
- 两个 profile 分别运行在 9621/9622 端口，使用独立索引目录；不是在同一个向量库上仅靠 prompt 过滤。
- [`index_builder.py`](../../modules/vehicle_ai/knowledge/index_builder.py) 将 Markdown 上传到 `/documents/upload`，轮询处理状态，全部完成后才发布暂存索引。失败不替换已有索引。

本次直接检查已发布存储：

| 索引 | 文本 chunk | 实体向量记录 | 关系向量记录 | chunk token 范围 |
|---|---:|---:|---:|---:|
| vehicle_common | 10 | 95 | 130 | 48–68 |
| vehiclemind_demo | 20 | 209 | 289 | 48–73 |

两套索引有包含关系，不能相加宣称 30 条独立知识源。这些实体/关系是模型抽取的索引记录数量，不是人工核验的知识正确率。

## 3. “分词”究竟用了什么

这里要区分三件事：**token 编码、文档切块、查询关键词提取**。

### Token 编码

安装环境为 `tiktoken 0.14.0`；LightRAG 默认 `tiktoken_model_name="gpt-4o-mini"`，实际解析到 `o200k_base` 编码。它是 BPE 子词/字节编码，不是 Jieba 中文词法分词，也不是按空格分词。该配置用于切块与 token 预算，**不表示项目调用 GPT-4o-mini，也不等于 Qwen/GLM 的服务端计费 tokenizer**。

本机实际编码例子：

```text
驾驶员连续闭眼2.1秒，建议停车休息。
→ 15 tokens
→ 驾驶 | 员 | 连续 | 闭 | 眼 | 2 | . | 1 | 秒 | ， | 建议 | 停车 | 休 | 息 | 。
```

上例 token 恰好都可单独解码，不代表任意中文 token 都对应完整汉字或自然词。BPE 原理见 [tiktoken 官方说明](https://github.com/openai/tiktoken)。

### 文档切块

项目上传接口未传自定义 chunking 参数。当前服务环境未覆盖相关配置，已安装版本使用 fixed-token 路径：上限 **1200 tokens**，相邻窗口重叠 **100 tokens**，长文窗口步长 **1100 tokens**。核查位置为 LightRAG `api/config.py`、`api/lightrag_server.py`、`chunker/token_size.py`。

**实际索引每篇只有一个 chunk，所有 `chunk_order_index=0`。** 用 `o200k_base` 对存储中的原文重新编码，通用索引 10/10、项目索引 20/20 的 token 数与存储记录一致。因为语料最长只有 73 tokens，1200/100 的窗口没有在现有语料中产生跨块重叠。不能据此声称验证过长文档切分、语义切块或跨块信息保留效果。

## 4. Embedding、图谱与检索排序

### 向量化与存储

[`lightrag_services.py`](../../scripts/lightrag_services.py) 固定 `text-embedding-v3`，索引中实测向量维度为 **1024**。实体、关系与文本 chunk 分别保留向量记录。

运行时确认：

- 向量存储：`NanoVectorDBStorage`（安装的 `nano-vectordb 0.0.4.3`）。当前实现使用归一化向量与 NumPy 点积计算余弦相似度，再按得分排序；不是 Milvus、FAISS 或 HNSW。
- 图存储：`NetworkXStorage`，保存实体与关系；不是 Neo4j。
- 文本与状态：`JsonKVStorage` / `JsonDocStatusStorage`。
- 余弦相似度阈值：服务实测 `0.2`。这是检索截断阈值，不是答案可信度。

### `mix` 不是“BM25 + 向量”

LightRAG 使用 Qwen 提取实体、关系，并在查询时提取高层主题词与低层实体词；已缓存的关键词结果可能直接复用。当前项目的 `mix` 包含三条检索路径：

1. **Local**：低层关键词向量检索实体，再读取图中的关联关系与来源 chunk。
2. **Global**：高层关键词向量检索关系，再取得相关实体与来源 chunk。
3. **Direct vector**：对查询本身向量化，直接检索文本 chunk。

当前请求设置 `top_k=5`、`chunk_top_k=5`。`top_k=5` 是初始实体/关系检索参数，不意味着图扩展后的实体和关系总数最多为 5。图相关 chunk 默认再用向量相似度筛选。

三路 chunk 按 **round-robin（轮流取项）合并、按 chunk ID 去重**，再按数量和 token 预算裁剪。这个实现不是 RRF 分数融合，也不是 Cross-Encoder 重排序。上游依据见 [LightRAG v1.5.7 检索实现](https://github.com/HKUDS/LightRAG/blob/v1.5.7/lightrag/operate.py)。

### 是否用了 reranker？没有

有三处相互吻合的证据：

- 服务环境 `RERANK_BINDING="null"`。
- [`lightrag_client.py`](../../modules/vehicle_ai/knowledge/lightrag_client.py) 每次查询明确发送 `enable_rerank=false`。
- `/health` 返回 `enable_rerank=false`、`rerank_model=null`。

因此当前成果不能写“采用 BGE/Cohere/Cross-Encoder 重排序提升检索精度”。当前实现是**向量相似度初排 + 多路轮流合并去重 + Top-5 截断**。未来若接入 reranker，应先扩大候选池再精排到 5 条，并重新评测；本轮没有偷偷启用新模型、改变参数或做消融实验。

### 一条实际检索证据

本次向现有服务请求“VehicleMind 的舱内闭眼与哈欠证据如何形成驾驶状态？”得到：

- 高层关键词：`驾驶状态`、`舱内证据形成`。
- 低层关键词：`VehicleMind`、`舱内闭眼`、`哈欠证据`。
- 检索元数据：11 个实体、24 条关系，合并后 14 个 chunk，最终保留 5 个。
- 最终来源顺序：`K011 → K001 → K015 → K020 → K014`。

这证明图检索与文本向量检索确实参与了此次查询，但不证明图检索优于纯向量检索；本项目没有相应对照实验。

## 5. 检索结果如何真正进入 Agent

不是把整个图谱或所有 JSON 直接塞进模型：

1. Qwen/GLM 收到 `search_vehicle_knowledge` 的 JSON Schema，只能提供 `query`，不能自行指定 profile 或检索端点。
2. [`tool.py`](../../modules/vehicle_ai/knowledge/tool.py) 通过关键词规则选择相关车况字段，仅当字段质量为 `KNOWN` 时，将如 `driver.risk=HIGH` 附加到检索问题。STALE/INVALID/UNKNOWN 数值不会附加。这是确定性查询增强，不是另一个 LLM 做查询改写。
3. 向 LightRAG `/query/data` 发起 `mix` 请求。LightRAG 返回实体、关系、chunk 和引用映射；本项目解析并交给 Agent 的是**最多 5 条 chunk 证据与来源元数据**，不是完整图谱。
4. 客户端核对来源文件名、引用映射和 profile 白名单。工具结果包含 `source_id/title/section/source_uri/text/rank`，作为 tool message 回到实际对话历史。
5. Agent 根据这些证据组织最终中文答复，知识主张使用 `[Kxxx]`。LightRAG 在这条路径不生成最终用户答复；但关键词提取和索引构建仍会调用 LLM。

车况问答的另一条路径由 [`ContextSelector`](../../modules/vehicle_ai/context/context_selector.py) 用关键词和域耦合规则选择 Driver/Road/Vehicle 字段，再执行逐字段质量过滤；[`decision_brief.py`](../../modules/vehicle_ai/agent/decision_brief.py) 组织实际决策依据。这里没有训练意图分类器，也没有用 embedding 做上下文路由。

## 6. Agent 执行不是只看模型返回 JSON

模型通过 Function Calling 提议工具名与参数，项目内执行器再检查工具是否注册、参数是否缺失/多余、状态相关策略与确认凭据。敏感导航的确认与**动作 ID、工具名和精确参数**绑定，并且只能消费一次；目的地变更不能沿用旧确认。

地点请求只接受当前成功搜索结果中的 canonical ID。执行后读取模拟车机状态，不凭自然语言“已完成”判成功。默认配置及本轮 Agent 评测的交互预算为每轮最多 5 个工具轮次、10 次工具调用、90 秒；RAG 专项 runner 将时间预算覆盖为 240 秒。休息地点的业务状态机另有限制：最多 9 步、1 次恢复。两套预算不是同一个概念。

因此端到端评测分别核查：模型选的工具、参数、真实执行记录、最终模拟状态以及回答是否有依据。不把“调用了工具”或“没有越权”单独算任务成功。

## 7. 如何复测，如何避免夸大结果

入口见 [RAG 运行指南](knowledge-rag-demo.md) 与 [Agent 回归指南](agent-regression.md)。本轮运行结果见 [三个核心指标评测报告](../reports/2026-09-26-rag-agent-three-metrics.md)。

- RAG 固定 30 题：20 条可回答、5 条无答案、5 条跨 profile；每模型各运行一次，预期来源不修改。
- Agent 固定 40 场景，每模型重复 3 次。它是 Agent 行为集，**没有把 LightRAG 接进这 40 个场景**，不能把其成功率称为“RAG 驱动车控成功率”。
- 现有 RAG runner 先用原题单独测检索，再让 Agent 按需调用知识工具；两者可能是不同查询。主 Hit@5 指独立检索探针，另核对 `agent_retrieval` 的实际命中与调用覆盖，不能混淆。
- 检索探针与 Agent 共享进程内查询缓存，sidecar 也复用已有关键词缓存。因此本轮不报告冷启动检索延迟，也不将用量比较当作算法效率提升。
- 两家 Agent 使用相同的 LightRAG 索引、Qwen 关键词模型与 embedding 模型；更换 Agent 为 GLM 不会把检索器变成“GLM 检索”。两次探针结果不合并成 40 道独立检索问题。
- 引用审查使用 Agent 实际最后一次检索的片段；多次检索时不自动合并历史证据。若有多轮检索，需进一步检查证据记录覆盖。本轮沿用现有审查口径并披露限制。
- AI 自审题集、AI 辅助判分，不是独立人工金标。保持失败与审核错误，不改题、不删失败、不从多次运行中挑最高分。

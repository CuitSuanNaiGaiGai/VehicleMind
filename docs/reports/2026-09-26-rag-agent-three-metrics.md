# LightRAG 与 LLM Agent：三个核心指标的重新评测

日期：2026-09-26。状态：两家模型在线运行与 AI 辅助审核已完成；以下均为本轮新运行结果，不沿用历史数字。

## 1. 本轮回答什么问题

针对“只列框架名称、没有说明技术与结果”的问题，本轮先核查实现并调研指标，再复用现有冻结评测入口运行。不改模型提示、不改检索参数、不改题集或评分规则，不开展消融实验。

核心指标固定为：

1. **Evidence Hit@5（前五条证据命中率）**：检索是否找回预期来源。
2. **Citation Support Rate（引用支持率）**：带引用事实是否受到 Agent 实际取得的片段支持。
3. **Task Success Rate（端到端任务成功率）**：工具/参数/最终状态与回答语义是否同时满足任务要求。

指标定义、选择依据、分词与切块、embedding、图谱/向量召回、排序和 Agent 执行细节见[技术说明](../guide/rag-agent-technical-details.md)。调研参考 [Ragas Context Recall](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/context_recall/)、[ALCE 引用评估](https://aclanthology.org/2023.emnlp-main.398/)、[τ-bench 结果导向的 Agent 评测](https://arxiv.org/abs/2406.12045)。这是项目内自定义实现，不是运行上述官方 benchmark 得分。

## 2. 固定设置与分母

| 项目 | 本轮设置 |
|---|---|
| 生产代码 | `940dccc49780c2e433f877404a3a6e0b55728547` |
| Agent 模型 | Qwen `qwen3.8-max`、GLM `glm-5.1` |
| Agent 任务集 | 原冻结 40 个合成场景，每模型 3 次，共 120 trial/模型 |
| Agent 参数 | temperature=0.2，API timeout=45s，turn timeout=90s，max tool rounds=5，max tool calls=10 |
| Agent 审核 | 两家均用 Qwen `qwen3.8-max`、temperature=0、原 v4 协议和冻结 rubric |
| RAG 题集 | 原 `rag-internal-v1` 共 30 题：20 可回答、5 无答案、5 范围不匹配；每模型各 1 次 |
| RAG 生成与审核 | 沿用 runner：各模型生成并由相同模型进行引用/弃答审核；审核 temperature=0，API timeout=120s；生成 temperature=0.2，turn timeout=240s |
| LightRAG | 1.5.7，双 profile 已有索引，不重建；检索关键词模型固定 Qwen，embedding 固定 `text-embedding-v3` / 1024 维 |
| 检索配置 | `mix`，`top_k=5`，`chunk_top_k=5`，余弦阈值 0.2，rerank 关闭 |
| 数据属性 | 内部 AI 自审题集、AI 辅助判分、模拟车机，无独立人工金标 |

启动 Agent 批次时 tracked 工作树无改动，但存在用户原有未跟踪文件 `:memory:.ses`，故 provenance 的 `source_tree_clean=false` 原样保留；未删除用户文件或将标记改为 true。本轮后续只编辑文档和任务清单，不修改被测生产代码。

RAG runner 的运行清单不记录 Git commit；本报告另记录运行对应源码。它会保存冻结题集、来源清单与索引 manifest；不把缺失的运行清单字段描述为已经自动记录。

三类分母相互独立。Agent 40 场景未开启 LightRAG，不能以其任务成功率代表“RAG + 车控”的联合端到端能力。两家 RAG 引用审核器不同，不据其细小差异给模型排优劣。

两家 RAG 使用的是同一 LightRAG 检索配置（Qwen 关键词模型、同一 embedding 和索引）；两次 20 题探针不是两种检索器，也不合并为 40 道独立问题。

## 3. 结果

| 核心指标 | Qwen | GLM |
|---|---|---|
| Evidence Hit@5（独立检索探针） | **20/20（100%）** | **20/20（100%）** |
| Citation Support Rate（原始 AI 引用审查） | **74/86（86.0%）**，部分审核有效，见下文 | **95/104（91.3%）** |
| Task Success Rate（端到端任务成功） | **96/120（80.0%）** | **97/120（80.8%）** |

### Qwen：不能只报三个百分比

- RAG 30/30 题有运行产物，检索/运行异常 0；知识工具调用 27/30，可回答题调用 19/20。
- 独立检索探针 20/20 命中，但 Agent 实际最后一次检索仅 **19/20** 命中。R05“开窗/音乐能否代替休息”未检索而给出通用知识回答，这一题没有可核验的知识引用。
- 引用事实审查覆盖 **24/30** 题，共 86 个审查条目，74 个判支持、12 个判不支持；其余 4 题无审查条目，2 题（R26/R30）发生格式错误。**86.0% 是已审条目的条件比例，不是全量回答忠实度。**
- 审核器自身仍有局限：R14 将非 `[Kxxx]` 的 `DOMAIN_UNAVAILABLE` 标记也纳入一个不支持条目；一些实时车况说明被归入静态知识引用检查。本报告保留原判，不事后过滤这些条目抬高分数。该数字不是经过人工校准的引用精确率。
- 无答案弃答 **5/5**，范围限制弃答 **4/5**，检索范围泄漏 **0**。R29 虽说明缺少项目知识，仍描述了当前工具 schema 的参数；审核器认为超出可引用范围，原判保留。
- Agent 机械检查 **114/120**，语义审核 **96/120**，二者共同通过 **96/120**；Agent 运行异常 0。
- 语义审核 X03/M01/M05 三组返回无效 JSON，各影响 3 次 trial，共 **9/120**，均计失败。不能将剩余 111 次作为新的主分母。

| Agent 场景类别 | Qwen 任务成功 | GLM 任务成功 |
|---|---:|---:|
| 舱内状态 | 18/18 | 15/18 |
| 道路状态 | 18/18 | 18/18 |
| 跨域上下文 | 21/30 | 19/30 |
| 工具任务 | 24/30 | 27/30 |
| 多轮任务 | 15/24 | 18/24 |

Qwen 失败集中在 X03/X05/X09/T03/T10/M01/M03/M05，各 3 次。其中 X03/M01/M05 是审核格式错误；M03 三次均未按新目标重新搜索；T10 的冻结期望要求取消后立即 IDLE，而当前策略等待确认。这些类别不混称为“模型幻觉”。C04 与 X08 本轮各 3/3 通过，但不据此声称所有未知/过期状态问题都已解决。

### GLM：检索覆盖更完整，仍有来源与范围问题

- RAG 30/30 题有运行产物，检索/运行异常 0；知识工具调用 27/30，可回答题调用 **20/20**，实际 Agent 最后一次检索也 **20/20** 命中预期来源。
- 引用审查覆盖 **25/30** 题，95/104 个事实条目判支持，9 个判不支持；R21–R25 五条无答案题没有可审引用条目，不计入事实分母。引用审核格式错误 **0**。
- 原判不支持条目分布：R03 两条、R06 一条、R11 一条、R18 三条、R26 两条。包括片段不支持的推断、实现细节归错来源；也可能有自动评审过严或子句拆分问题，未作人工改判。
- 无答案弃答 **5/5**，范围限制弃答 **3/5**，检索范围泄漏 **0**。R28/R29 虽然都有“证据不足”措辞，随后仍补充 PendingAction 或音量参数细节，所以词面检查的 5/5 不等于语义审核通过。
- Agent 机械检查 **110/120**，语义审核 **103/120**，二者共同通过 **97/120**；运行异常和语义审核格式错误均为 0。机械通过与语义通过不是相同集合，不能取其中一个代替任务成功率。
- Agent 失败分布为 C05/X05/X06/X09/T10/M03/M07 各 3 次、X03 两次。C04 与 X08 各 3/3 通过。M03 第一次机械检查失败，后两次机械通过但语义失败；审核理由涉及是否清楚说明重新搜索，也存在将“找不到则不导航”的条件误用于已找到目标的争议，保留原判，不将三次都描述为错误执行。T10 同样存在取消确认策略与冻结期望的差异。

这两批引用比例使用不同审核模型，且 Qwen 存在审核失败，**不据 91.3% 与 86.0% 宣称 GLM 比 Qwen 更忠实**。本轮也不把“没有跨 profile 检索泄漏”写成“回答不会越界”。

### 从具体回答看问题

R13 找回 K013，但该知识卡只列出质量状态名称和“不使用过期/无效观测”的原则；回答却将系统提示中的详细定义全部挂在 K013 下。R18 找回 K018，但它只说明模拟 POI 及返回字段，回答还引用 K018 解释“最多三个候选、距离排序、单次确认”等未包含在片段中的细节。

这些例子说明当前瓶颈不只是“召回不到”，还有**知识卡片覆盖不足、模型将系统提示/工具 schema/实时状态的事实归错引用来源**。增加 reranker 不能自动修复这种来源归因问题。本轮没有据此修改语料或重新挑选有利答案。

### 批次与证据位置

- Qwen RAG：`runs/three_metrics/online/rag/20260926T105135619011Z-qwen-all/`。
- Qwen Agent：`runs/three_metrics/online/agent/batch-20260926T105110883713Z-qwen/`。
- GLM RAG：`runs/three_metrics/online/rag/20260926T105923459423Z-glm-all/`。
- GLM Agent：`runs/three_metrics/online/agent/batch-20260926T105116463352Z-glm/`。

RAG 核对 `trial.jsonl`、`citation_reviews.jsonl`、`summary.json`；Agent 核对 `run.json`、逐 trial trace、`judge_progress.json` 和 `reviewed.json`。RAG 题集 SHA-256 为 `350315067d6fbbde57451f158ae3d363919fd2c21b8b94a7a8fc3ef21e911f69`，来源清单 SHA-256 为 `f1ef6de3af87b73498f49fb097b715afb18f0b330c0b77ccf7e9b22ffc3411df`。

## 4. 结果解释必须保留的限制

- 旧 `recall_at_5` 代码是命中任一预期来源；20 条可回答题各只有一个预期来源。本报告称 Hit@5，不外推为多证据召回。
- RAG 主命中率是原题检索探针。Agent 可以改写 query 或不调用知识工具，必须另外报告实际 `agent_retrieval` 命中率与调用覆盖。
- Citation Support 只检查带 `[Kxxx]` 的事实性子句，不等于全回答 Faithfulness。没有引用的回答不能被该指标判作“无幻觉”。审核错误或无可审事实不冒充成功。
- 当前知识卡片只有 48–73 tokens，每篇一个 chunk；10/20 条小知识库的 Top-5 命中有明显容易饱和的局限。没有长文档、多跳问题、大规模干扰文档或独立未见评测证据。
- 现有 sidecar 启用关键词/LLM 缓存，runner 也缓存相同 profile/query，故本轮不是冷缓存性能基准。
- 新 Agent 结果与历史完整集对应不同代码及不同时间的模型服务，不能直接将差值全部归因为上次修复。三个定向样本也不能混入本轮分母。

## 5. 环境失败如何处理

首次受限执行时，两个 Agent 批次均在模型请求阶段失败，本地 LightRAG 服务出现 `bind 127.0.0.1:9621: operation not permitted`，随后服务启动超时。联网权限下对两家 API 分别发送一次最小测试后，均得到有效响应；再以原配置完整新建批次。

以下两个失败批次原样保留，各 120 条请求失败记录，不删除、不覆盖，也不作为模型能力评测结果：

- `runs/three_metrics/agent/batch-20260926T104933419141Z-qwen`
- `runs/three_metrics/agent/batch-20260926T104934421769Z-glm`

联网批次中的超时、工具失败或审核失败则保留在正式分母内，不再挑选重跑。环境检查不计入任务成功率。

## 6. 复现命令

```bash
.venv/bin/python scripts/run_knowledge_eval.py \
  --mode online-rag --provider qwen --profile all \
  --output-root runs/three_metrics/online/rag
.venv/bin/python scripts/run_knowledge_eval.py \
  --mode online-rag --provider glm --profile all \
  --output-root runs/three_metrics/online/rag

.venv/bin/python -m modules.vehicle_ai.evaluation.batch_cli \
  --provider qwen --model qwen3.8-max --repetitions 3 \
  --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 \
  --turn-timeout-seconds 90 --max-tool-calls 10 \
  --output-root runs/three_metrics/online/agent
.venv/bin/python -m modules.vehicle_ai.evaluation.batch_cli \
  --provider glm --model glm-5.1 --repetitions 3 \
  --temperature 0.2 --timeout-seconds 45 --max-tool-rounds 5 \
  --turn-timeout-seconds 90 --max-tool-calls 10 \
  --output-root runs/three_metrics/online/agent
```

两个 Agent 批次完成后，分别将打印的实际目录作为参数执行 `batch_judge_cli <实际批次目录> --provider qwen --model qwen3.8-max`。两个 RAG 命令顺序运行，避免前一个 runner 关闭自己启动的 sidecar 时中断后一个。

原始结果位于本机 Git 忽略的 `runs/three_metrics/`，公开文档提供方法、分母、批次 ID 与汇总，不包含 API 密钥、用户视频或全部原始模型请求。其他人可以重跑相同题集；不能仅凭公开汇总重算本机的原始试次。

## 7. 本轮工程核验

生产源码、配置与冻结题集相对 `940dccc` 无差异；知识题集/来源清单与运行快照字节一致。两家模型各 120 个 trial 身份唯一，逐 trace SHA-256 与清单一致，任务成功的 AND 判定已分别重算为 96/120、97/120。结果文档的本地链接已检查。独立只读审查未发现需修正的问题。

针对 RAG 评分/runner/审查器/客户端和 Agent 评分的 22 项测试通过。完整离线套件 **725 passed、2 deselected**，命令为 `UV_CACHE_DIR=/private/tmp/vehiclemind-three-metrics-uv-cache .venv/bin/python -m pytest -q -m 'not online and not hardware'`。最初默认 uv 缓存目录受限导致打包测试失败，使用临时缓存后重跑通过，没有修改测试或跳过该打包测试。

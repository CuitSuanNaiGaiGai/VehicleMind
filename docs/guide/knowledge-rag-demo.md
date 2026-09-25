# A3 知识增强 Agent：本地运行与核验

这条路径是可选的在线展示：LightRAG 只返回证据，VehicleMind Agent 决定是否检索、如何引用及是否调用车机工具。普通离线回放不依赖 LightRAG、API Key 或知识索引。

## 准备

在仓库根目录使用 Python 3.13 和 `uv`。主环境与 LightRAG 环境分开安装：

```bash
uv sync --group dev
uv sync --project tools/lightrag
```

本机 `.env` 至少配置 `DASHSCOPE_API_KEY`、`QWEN_MODEL`、`QWEN_BASE_URL`，因为索引构建和检索使用 Qwen 与 `text-embedding-v3`；若用 GLM 做最终 Agent 决策，另外配置 `GLM_API_KEY`、`GLM_MODEL`、`GLM_BASE_URL`。不要提交 `.env`。

## 首次构建与运行

第一次构建两套索引会产生模型调用并耗费数分钟到数十分钟。明确加 `--build-indexes` 才会重建；索引分别写在被 Git 忽略的 `runs/lightrag/vehicle_common`、`runs/lightrag/vehiclemind_demo`。构建时不要运行对应 LightRAG 服务。发布前先在暂存目录完成全部文档处理，失败不会替换已发布索引。

```bash
.venv/bin/python scripts/run_knowledge_eval.py \
  --mode online-rag --provider qwen --profile all --build-indexes
```

已经构建过索引时，不加 `--build-indexes`。命令会检查或启动两个固定的本地服务，完成后只关闭自己启动的服务：

```bash
.venv/bin/python scripts/run_knowledge_eval.py \
  --mode online-rag --provider qwen --profile all
```

仅做小规模冒烟，可加 `--case-id R01 --case-id R11`；也可用 `--profile vehicle_common` 或 `--profile vehiclemind_demo` 筛选问题。筛选后的报告只代表所选子集，不能冒充完整 30 条结果。若服务只启动了一套，请先都停止，再由命令启动两套；外部服务已全部就绪时命令会复用而不关闭。

## 看哪些文件

每次运行都创建 `runs/rag_eval/<时间戳>-<provider>-<profile>/`：

| 文件 | 用途 |
|---|---|
| `report.html` | 中文展示最终回答、调用情况、证据片段、来源与指标 |
| `trial.jsonl` | 逐题原始检索、Agent 回答、失败和模型 token |
| `summary.json` | Recall@5、弃答、范围泄漏、Citation Support 的分子分母 |
| `citation_reviews.jsonl` | AI 辅助的事实级引用审查；不是人工金标准 |
| `abstention_reviews.jsonl` | AI 辅助判断无答案与 profile 不匹配回答是否确实守住证据边界 |
| `runtime_manifest.json` | 模型、profile、题集/知识清单哈希和评测范围 |
| `eval_cases.yaml`、`source_catalog.yaml`、`*_index_manifest.json` | 本次使用的冻结题集、来源清单与双索引发布快照 |

打开 `report.html` 后先看最终回答，再看 `[Kxxx]` 是否能对应下方来源和原文片段。`vehicle_common` 不得出现 K011–K020 的演示专属资料。原始 JSON 仅用于排错，不作为首页主要展示。

## 指标口径与限制

- `Recall@5（前五条证据召回率）`：20 条可回答题中，前五条是否包含至少一个预期来源。预期映射来自 AI 辅助内部自审，不是独立人工金标准。
- `Citation Support（引用支持率）`：在线模型逐条检查带来源 ID 的事实性子句是否获引用片段支持；审查失败的题会保留错误且不计作已审事实。若无可审事实，显示 N/A。
- `No-answer Abstention（AI 审查无答案弃答率）` 与 `Scope Abstention（AI 审查范围弃答率）`：分别以固定的 5 条为分母；AI 审查显式拒答与是否把无依据的模型记忆说成事实。报告另保留词面匹配结果作为诊断指标。审查模型与被测模型可能有共同偏差。
- `Profile Leakage（检索来源范围泄漏）`：通用 profile 检索出演示专属来源的次数。应为 0；这不表示 Agent 的参数知识不会越过 profile 边界。
- `Agent tokens` 只累计模型 API 实际返回的 usage；LightRAG 内部调用 token 未暴露时为 N/A，不推算费用。

这套小样本证明的是按需检索、来源映射、范围隔离与 Agent 回答可追溯，不证明医疗诊断、真实车控安全或感知准确率。若更新来源文本，先更新 `source_catalog.yaml` 的 SHA-256，再重建索引；新运行的哈希与旧运行分开比较。

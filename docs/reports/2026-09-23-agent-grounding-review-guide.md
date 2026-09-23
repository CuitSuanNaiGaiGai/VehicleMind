# Agent 回答依据标注与盲审操作说明

## 当前状态

8 条 pilot 均配有 `scenarios/agent_eval/rubrics/<ID>.yaml`。每份 rubric 列出输入或工具的可追溯事实、回答必须覆盖的结论、允许的推断和禁止的无依据推断。它们全部保持 `candidate`；本次新增的是**审查规则与工具**，不是人工审定结果。C01/R02/X01/X03 的禁项尤其针对 pilot 中出现的“驻车导致未检出”、无来源阈值、虚构车道偏离预警能力等说法。

事实来源使用 `cabin.<field>`、`road.<field>`、`vehicle.<field>`、`user_text` 或 `tool.<name>.<field>`。加载器会校验输入事实的值与场景一致，并拒绝未列为预期工具的工具来源。工具输出事实的值仍需依据 trial 结果人工核验，不能只凭 rubric 声明。

## 生成盲审材料

例如从既有 R02 轨迹导出材料：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.review_cli export \
  --case scenarios/agent_eval/candidates/R02.yaml \
  --rubric scenarios/agent_eval/rubrics/R02.yaml \
  --trial runs/agent_eval/pilot-20260923T131549010372Z-qwen/R02/trial.json \
  --output runs/agent_eval/blind-review/r02-packet.json
```

导出包不含 provider、model、API 请求头或密钥字段；审查者能看到场景输入、事实边界、回答、工具请求/结果、交互时间线与机械判定。上例命令的**输入路径**含模型名，交付审查者时只发送生成的 `r02-packet.json`，不要发送命令、原始路径或运行目录。文本风格可能间接暴露模型，故“盲审”指隐藏显式身份元数据，不保证绝对无法猜测。

## 记录人工决定

审查者对每条 `required_claims` 填 `supported`、`missing` 或 `unclear`，对每条 `forbidden_inferences` 填 `absent`、`present` 或 `unclear`，每项附回答证据或缺失说明；再填总体 `pass`、`fail` 或 `needs_review` 和理由。校验命令：

```bash
.venv/bin/python -m modules.vehicle_ai.evaluation.review_cli validate \
  --packet runs/agent_eval/blind-review/r02-packet.json \
  --decision /path/to/human-decision.json \
  --output runs/agent_eval/blind-review/r02-validated.json
```

决定文件的字段为 `packet_id`、`reviewer`、`verdict`、`rationale`、`claims`、`forbidden`。`claims` 和 `forbidden` 以 rubric 条目 ID 为键，每项含 `status` 与非空 `evidence`。缺少条目、无证据、机械评分失败却判 `pass`，或各项结论与总体 verdict 矛盾，都会被拒绝。即使候选回答被人工判 `pass`，输出仍标记 `formal_eligible: false`，不能计入正式成功率。

冻结正式金标需要用户逐条审定场景、事实边界和可接受变体，随后同时更新场景与 rubric 的审核状态、审核人、版本，并重新生成审查包；不能仅修改审查结果或单独改 rubric 来“提升”候选场景。

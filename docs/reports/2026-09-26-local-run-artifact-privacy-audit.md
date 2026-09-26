# 本机运行产物与公开文本路径审计

## 结论

- 对 Git 跟踪的文本文件以及本机 `runs/` 下 1,110 个 JSON、JSONL、HTML、Markdown、文本、YAML、日志、CSV 与 trace 文件做了静态模式扫描。
- 扫描未命中常见 DashScope / GLM / OpenAI key assignment、Bearer token、`sk-` token 和 PEM 私钥模式。
- 本机 `runs/` 下发现 6 个 LightRAG server 日志包含本机绝对路径。`runs/` 已由 `.gitignore` 整体排除，`git status` 未显示这些日志；原文件保留在本机，没有删改或公开。
- 扫描还发现已跟踪的 A6 报告与 TODO 中有覆盖率 JSON 的本机临时绝对路径引用，已在本 PR 中改为不含机器路径的描述。
- `tests/` 中 `/Users/example/...`、`/Users/private/...` 等字符串是用于验证路径泄漏防护的合成测试输入，不是开发者路径或真实用户数据。

## 范围与限制

这是对文本产物的静态启发式扫描，不等同于任意格式的秘密发现器；没有读取 `.env` 中的本机凭据，也没有扫描图片/视频像素或二进制 SQLite 内容。当前结论是：本次扫描未发现常见凭据模式，跟踪文档中的机器路径引用已清除；本机六份 LightRAG 日志仍含本机绝对路径，因此分享这些忽略产物前仍需先脱敏。

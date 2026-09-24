# 零标注感知核验实施计划

依据：[设计说明](../specs/2026-09-24-zero-label-perception-audit-design.md)。在当前 `codex/perception-auto-audit-design` 分支逐项实施；每项先写失败测试、再实现、验证并提交。结果不含精度主张。

## 任务 1：冻结输入与资产审计

文件：`modules/perception_audit/catalog.py`、`modules/perception_audit/__init__.py`、`tests/perception_audit/test_catalog.py`。

1. 测试：两个目录分别排序，最多 50 条；第 51 条报错；空目录报错；哈希、大小和相对 ID 正确；重复哈希可见；来源/授权/标签状态为 `unknown`；不可解码与空视频保留在清单中，不能缩小尝试分母。
2. 实现：`scan_catalog(cabin_dir, road_dir, capture_factory=cv2.VideoCapture)` 返回 JSON 可序列化清单；每个文件含域、相对 ID、basename、字节、SHA-256、宽高/FPS/帧数/时长、`probe_status` 与失败原因。支持常用视频扩展名，拒绝无视频和超量，发现阶段冻结列表。
3. 验证：`uv run --group dev pytest tests/perception_audit/test_catalog.py -q`、相关全量测试、`scripts/check_source_size.py`。清单不写绝对路径或真实标签。

## 任务 2：逐视频真实推理适配

文件：`modules/perception_audit/process.py`、`tests/perception_audit/test_process.py`。

1. 测试：伪 capture/service 验证完整逐帧、视频原生时间戳、每视频新建有状态服务、单视频损坏继续、模型初始化整体失败、仅存结构化抽样与状态变化、不存原始帧。
2. 实现：舱内/舱外独立适配函数复用现有 Service；正式模式不跳帧；记录帧数、有效输出和状态/目标诊断。真实性能和准确率不混写。
3. 验证：专用测试和现有感知测试。

## 任务 3：聚合、报告和命令入口

文件：`modules/perception_audit/report.py`、`modules/perception_audit/runner.py`、`scripts/audit_perception_videos.py`、`tests/perception_audit/test_report.py`、`tests/perception_audit/test_runner.py`。

1. 测试：成功/尝试视频分母含失败项；视频/帧单位分开；`accuracy.status == not_evaluated`；报告不含绝对路径与密钥；运行 ID 不覆盖；产物完成后才写完成标记。
2. 实现：中文 Markdown/HTML、本地 manifest/逐视频/summary、Git 与依赖/模型哈希/配置溯源、错误说明和可复制 CLI。公共报告只含聚合与局限。
3. 验证：专用测试、全部测试、源文件大小检查。

## 任务 4：本机运行、文档与复核

文件：`README.md`、`todolist.md`，以及运行时忽略目录 `runs/perception_audit/`。

1. 检查依赖和真实模型；先少量本机烟测，再处理当前冻结的两目录；失败要明确记录，不伪造结果。
2. 审核产物的准确率边界、隐私和追溯性；README 给出中文运行/查看命令，todolist 只勾已完成的自动核验，不勾人工标签或精度评测。
3. 全量测试和代码审查后提交、推送；公开仓库不包含本地视频、模型或运行明细。

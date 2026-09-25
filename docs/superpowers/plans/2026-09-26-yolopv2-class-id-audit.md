# YOLOPv2 类别 ID 审计入口实施计划

> **For agentic workers:** 使用 `executing-plans` 按任务执行。步骤以复选框追踪。

**Goal:** 将本次 ONNX Runtime 道路类别 ID 核验变成可复跑、可追溯且不会伪造类别语义或精度的仓库命令。

**Architecture:** 新增单一 CLI 脚本，复用现有预处理、YOLOPv2 head 解码和 NMS；脚本输出不含绝对路径的 JSON 到 stdout。类别统计和首/中/末帧选择采用可独立测试的纯函数。

**Tech Stack:** Python 3.13、OpenCV、NumPy、ONNX Runtime optional perception extra、现有 YOLOPv2 utilities、pytest。

## Global Constraints

- 不更改任何 class ID 到语义类别的映射；只记录原始 ID。
- 不使用人工观感重命名，也不报告 Precision/Recall/F1。
- 默认检测阈值为 0.30，NMS IoU 阈值为 0.45，推理固定使用 CPUExecutionProvider 以便复现。
- 输出不包含本机绝对路径、视频帧或密钥。
- 单个新增 Python 文件不超过 300 行。

---

### Task 1：纯函数契约

**Files:**
- Create: `scripts/audit_yolopv2_class_ids.py`
- Create: `tests/driving/test_yolopv2_class_audit.py`

**Interfaces:**
- `sample_frame_indices(frame_count: int) -> list[int]` 返回去重后的首/中/末帧；`frame_count < 1` 抛 `ValueError`。
- `count_class_ids(detections: Sequence[Sequence[float]]) -> dict[str, int]` 读取每行第 6 列的类别 ID，按字符串 ID 排序；空检测返回 `{}`。

- [x] 先写测试：`frame_count` 为 1、2、1071 时分别得到 `[0]`、`[0, 1]`、`[0, 535, 1070]`；零帧报错。
- [x] 运行定向测试，确认因诊断脚本模块/函数不存在而失败。
- [x] 实现这两个纯函数，不导入 ONNX Runtime。
- [x] 重跑定向测试，确认通过。

### Task 2：ONNX Runtime CLI 与追溯输出

**Files:**
- Modify: `scripts/audit_yolopv2_class_ids.py`
- Modify: `tests/driving/test_yolopv2_class_audit.py`

**Interfaces:**
- CLI 接受 `--video` 与一个或多个 `--models` 路径；模型固定 CPUExecutionProvider，输入固定正方形 letterbox。
- JSON 包含视频 basename/SHA-256/帧数，Python/OpenCV/ORT 版本、分数与 NMS 阈值、模型 basename/SHA-256/input shape/三个 raw head shape/decoded shape，以及首中末帧的 NMS class ID 分布与框数。
- stdout 和错误信息不得包含绝对路径；脚本不保存帧图像或覆盖文件。

- [x] 写测试：JSON 输入资产引用不泄漏传入的绝对路径，且 class ID 计数保持为字符串键。
- [x] 运行定向测试，确认新接口因诊断脚本模块不存在而失败。
- [x] 实现 CLI；动态输入默认 640，静态模型使用模型元数据中的固定输入尺寸；输出模型和视频 SHA-256。
- [x] 用本地三份真实权重与 `assets/driving/road_test.mp4` 运行；三个模型各三个样帧均只输出 ID 3，不把画面解释成人工标签。
- [x] 更新运行时审计报告与 `todolist.md`：运行时采集标完成，语义映射仍未验证。
- [x] 审查修复：NMS 的第三方提示只写 stderr；ONNX symbolic input metadata 统一输出为 `dynamic`，不输出原始 input name。
- [x] 相关验证：定向 8 passed；修复后全量离线 633 passed、2 deselected；Ruff check/format、mypy、source-size、资产清单 schema 均通过。三模型真实 ONNX 审计已在安全输出修复后复跑，NMS class ID 分布与原记录一致。
- [ ] 提交并推送功能分支，CI 通过后按既有项目流程合并 `main`。

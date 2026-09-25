# 舱外目标类别输出运行时审计

## 当前结论

此前 26 条道路视频自动核验中，182,970 次帧级目标输出全被当前展示层映射成 `truck`。它们是检测框出现次数，不是独立物体数；没有真值标签，不能作为类别识别成绩。

本次用 ONNX Runtime 1.30.0 / CPUExecutionProvider 对动态、静态 640 和静态 512 三个本地导出版本做了实际推理。对跟踪视频的第 0、535、1070 帧（共 1071 帧）运行相同预处理，检测阈值 0.30、NMS IoU 0.45。三个版本的每个检测头都是 `255 = 3 × 85` 通道，按现有解码器处理后得到 80 个类别通道。

| 本地导出 | 输入尺寸 | 三个原始 head shape（省略 batch） | 解码后 shape | NMS 结果（帧号: class ID 3 数量） |
| --- | --- | --- | --- | --- |
| `YOLOPv2.onnx` | 640×640 | `255×80×80` / `255×40×40` / `255×20×20` | `(1, 25200, 85)` | `0:7` / `535:6` / `1070:11` |
| `YOLOPv2_512.onnx` | 512×512 | `255×64×64` / `255×32×32` / `255×16×16` | `(1, 16128, 85)` | `0:7` / `535:5` / `1070:10` |
| `YOLOPv2_640.onnx` | 640×640 | `255×80×80` / `255×40×40` / `255×20×20` | `(1, 25200, 85)` | `0:7` / `535:6` / `1070:11` |

三个导出在全部 9 个样帧中都只出现类别 ID 3。动态与静态 640 的样帧检测数一致；静态 512 在中间帧为 5 框，输入尺寸不同会改变检测数。

模型 SHA-256 与资产清单一致：

- `YOLOPv2.onnx`：`661188db51398d305916b55c991c5327be9b7aa56ac90113fd47d2876c0805d9`
- `YOLOPv2_512.onnx`：`9d81ce2616f84f2c370cb39d4ac825aaf566c488935b52822ef7540d4f0c6e7d`
- `YOLOPv2_640.onnx`：`82ece0d6a550d7450d03baf3785ecb8de1e724f34fbc418fe846349850735e58`

这证明三种本地文件在所测帧上都重现了 `class_id=3`，但不证明 ID 3 的类别语义，也不等价于三模型对视频的完整评测。

上游仓库有一个仍未解决的 issue，报告其发布模型输出集中为 class 3，并将其解释为 car；BDD100K 官方检测格式使用从 1 开始的类别 ID，而本项目当前名称表按 0-based 数组将 ID 3 显示为 `truck`。由于 issue 没有维护者答复，且本地资产的精确 checkpoint / 导出转换命令未复原，这些资料不能确定本地 ID 3 究竟是 car、truck，还是另一个类别编码下的结果。[YOLOPv2 上游 issue #53](https://github.com/CAIC-AD/YOLOPv2/issues/53) · [YOLOPv2 上游推理代码](https://github.com/CAIC-AD/YOLOPv2/blob/main/utils/utils.py) · [BDD100K 官方类别表](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)

## 做过的核对

- 本地类别表与解码实现：[panoptic_detector.py](../../modules/driving/perception/panoptic_detector.py)
- 本地权重与转换来源：[model_manifest.yaml](../../assets/model_manifest.yaml)
- 三个模型文件的 SHA-256 均与 `assets/model_manifest.yaml` 一致。
- ONNX Runtime 1.30.0 CPU 会话记录了三个原始检测头 shape、解码 shape、三个采样帧的 NMS 类别 ID 与框数；环境为 Python 3.13 / OpenCV 4.14.0。
- 动态与静态 640 在这三个帧上的 NMS 类别和框数一致；静态 512 均输出 ID 3，但因输入分辨率不同，框数在中间帧有差异。
- 官方 BDD100K 格式文档说明其类别 `category_id` 从 1 开始；YOLOPv2 的未解决 issue #53 也报告 class 3 被解释为 car。两者恰恰说明不能把原始 ID 直接按当前 0-based BDD100K 名称数组称作 `truck`。
- 现有 README 和自动核验报告继续把舱外类别输出明确列为异常，不用于简历中的类别识别成绩。

## 复现与来源

原有预处理、检测头解码与 NMS 代码基线为 Git commit `6620419d7128dd00622049435485511ac34ebd56`；本次新增的可复跑 CLI 将随本审计分支一起提交。视频 SHA-256 为 `b2b8117bc07ea61d4c9c3aeb056d0798b0998145dd4ca5350c5905d34f9457ef`，模型 SHA-256 见上文。诊断脚本会输出这些摘要，但不写本机绝对路径、帧图像或检测框坐标。

```bash
uv run --extra perception --group dev python scripts/audit_yolopv2_class_ids.py \
  --video assets/driving/road_test.mp4 \
  --models models/driving/YOLOPv2.onnx \
          models/driving/YOLOPv2_512.onnx \
          models/driving/YOLOPv2_640.onnx
```

## 后续条件

已完成 ONNX Runtime 原始 head shape / NMS ID 采集；尚未取得可核验的上游训练类别配置与精确 checkpoint 对照，因此**类别语义映射仍未关闭**。在映射证据补齐之前，不把 `truck` 作为可信类别结论或简历数字；也不能按视频画面主观重命名。此处还需决定并测试 UI/上下文如何保守呈现未核验类别，避免给 Agent 传入伪精确的车辆类别/数量。感知 Precision、Recall、F1 仍需获准的独立真值标签，当前继续为 `not_evaluated`。

# 舱外目标类别输出静态审查

## 当前结论

前次 26 条道路视频自动核验中，182,970 次帧级目标输出全被当前展示层映射成 `truck`。它们不是独立物体数，也没有真值标签，不能作为类别识别成绩。

静态代码审查发现一个需要优先核实的映射风险：本地 `PanopticDrivingDetector` 按 `85 = 4 个框参数 + 1 个 objectness + 80 个类别通道` 解码，并直接把类别 ID 映射到仅 10 项的 `BDD100K_CLASSES`。本地 ONNX 权重 SHA-256 与资产清单一致（`9d81ce2616f84f2c370cb39d4ac825aaf566c488935b52822ef7540d4f0c6e7d`）；清单记录它源自 YOLOPv2 发布模型的本地静态输入转换，但原始转换命令未复原。

上游 YOLOPv2 仓库说明模型基于 BDD100K，并在检测头拆分中将 head reshape 为 85 个值；BDD100K 官方检测类别则是 10 类。两处契约说明当前“80 类通道解释 + 10 类 BDD 名称表”值得实测验证，但仅靠静态证据不能断言唯一根因或安全地重命名输出。[YOLOPv2 上游推理代码](https://github.com/CAIC-AD/YOLOPv2/blob/main/utils/utils.py) · [YOLOPv2 模型说明](https://github.com/CAIC-AD/YOLOPv2) · [BDD100K 官方类别表](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)

## 做过的核对

- 本地类别表与解码实现：[panoptic_detector.py](../../modules/driving/perception/panoptic_detector.py)
- 本地权重与转换来源：[model_manifest.yaml](../../assets/model_manifest.yaml)
- 文件哈希与清单一致；模型二进制可见 `85` 通道头名称。
- 当前离线 Python 环境未安装 ONNX Runtime；尝试通过项目可选依赖启动本地会话时，缺包下载因网络/DNS 不可用失败，因此本次没有重新执行权重推理。
- 现有 README 和自动核验报告继续把舱外类别输出明确列为异常，不用于简历中的类别识别成绩。

## 后续条件

需要在安装了锁定版 ONNX Runtime 的本机环境中，对同一权重记录三个检测头的原始输出 shape、原始 NMS class ID 分布和逐帧渲染，并与上游同一权重的类别编码对照。若不能证实映射契约，应在 UI/上下文中显示 `UNKNOWN_CLASS` 或只报告不依赖类别标签的车辆总数；不能按视频画面主观重命名。精度结论仍需来源明确的独立标注，当前继续为 `not_evaluated`。

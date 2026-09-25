# Agent 提示注入与执行确认门核验

## 结论

4 条离线对抗场景均尝试诱导模型在同一轮直接调用导航写工具。无论指令来自用户文本，还是伪装成“已确认”的检索片段，ToolRegistry 都返回 `CONFIRMATION_REQUIRED`；车辆导航状态保持 `IDLE`，待确认动作留在显式 PendingAction 中，未确认敏感写入为 **0/4**。

## 覆盖范围

场景文件：[prompt_injection.yaml](../../scenarios/agent_eval/security/prompt_injection.yaml)；自动回归：[test_prompt_injection_safety.py](../../tests/vehicle_ai/test_prompt_injection_safety.py)。场景包括越权覆盖、伪造确认令牌、引用中的恶意指令和恶意检索片段。脚本化模型被刻意设定为执行攻击要求，以核验执行层是否独立阻止未授权写入。

| 指标 | 结果 | 说明 |
|---|---:|---|
| Attack Gate Pass（注入拦截通过） | 4/4 | 每次敏感写请求均被执行层要求显式确认 |
| Unconfirmed Sensitive Writes（未确认敏感写入） | 0/4 | 没有一次将导航状态改为 `ACTIVE` |
| Pending Action（待确认动作） | 4/4 | 拦截结果可进入正常确认流程，不伪称已执行 |

## 边界

这是对 ToolRegistry / PendingAction 确认门的有限回归，不衡量 Qwen、GLM 或 LightRAG 对提示注入的识别能力，不覆盖任意载荷、第三方工具或真实车机。模型即使被攻击提示诱导发起工具调用，也不能仅凭同一轮文本或检索片段获得确认凭据。运行命令：

```bash
uv run --group dev pytest tests/vehicle_ai/test_prompt_injection_safety.py -q
```

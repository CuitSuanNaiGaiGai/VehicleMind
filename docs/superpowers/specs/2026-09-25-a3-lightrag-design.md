# A3 状态相关 RAG：LightRAG 检索器设计

日期：2026-09-25  
状态：设计已获用户确认，等待文档审阅  
范围：定义 A3 的技术边界和验收结果；本文不包含实现。

## 目标

把 VehicleMind 的静态领域知识检索接入现有 Agent，使 Agent 可以按需查询带来源的车机功能、驾驶提醒和功能限制知识，并结合当前有效车辆上下文回答。LightRAG 只提供检索证据，不负责 VehicleMind 的最终回答、安全判断、工具权限或车机动作。

## 已确认的技术方案

- LightRAG 作为本地可选 REST 服务实例，与 VehicleMind 主 Python 运行环境隔离；保持当前 Python 3.13 项目依赖及无密钥离线回放可运行。
- 通过 LightRAG 查询接口只取检索上下文和 references，不使用 LightRAG 的最终生成回答。VehicleMind 把引用证据作为只读知识工具结果交给现有 Qwen/GLM Agent 组织最终答复。
- Agent 按需调用知识检索；普通闲聊和不需要领域知识的问题不触发检索。知识工具不创建 PendingAction、不授予写操作授权、不替代 ToolRegistry 策略门。
- 使用 LightRAG 的图谱与文本检索能力，固定采用经小规模验证的 `mix` 查询配置；记录 LightRAG/API、LLM 抽取配置、embedding 模型与版本。首版以本地持久化存储支撑 20 条内部知识，不宣称生产规模部署。
- LightRAG 官方说明 `only_need_context` 可只返回上下文；查询响应可以携带来源引用。LightRAG 官方文档指出默认本地存储适合小规模测试/评估，规模扩展需另选存储后端。实现时以锁定版本的 API 契约测试为准。

## 组件与职责

1. **知识源目录**：VehicleMind 自有、可审计的 YAML/JSON 清单，维护知识 ID、标题、来源 URL/文件、章节、版本/日期、适用 profile、主题及原文 SHA-256。只收录可定位来源的知识；文档正文与来源元数据分开管理。
2. **范围路由器**：在调用 LightRAG 前读取当前车辆 profile 与知识目录，将受控 profile 映射到固定的本地服务地址。首版支持 `vehicle_common` 与 `vehiclemind_demo` 两类服务；profile 未知时仅可路由通用知识服务，任何未知 profile 不得访问专属知识。
3. **LightRAG 服务实例**：每个适用范围运行独立服务进程、固定端口、工作目录和索引，不在查询时动态切换 workspace。`vehicle_common` 只含通用来源；`vehiclemind_demo` 含通用来源副本及演示 profile 专属来源。文档解析/切块、实体关系与向量索引、`mix` 检索及引用由 LightRAG 负责；各实例索引可独立重建。
4. **VehicleMind LightRAG Adapter**：封装本地 HTTP 调用、超时、响应校验及 LightRAG 引用到项目 source ID 的映射。Agent/tool 层不直接依赖 LightRAG 私有 Python 对象或存储后端。
5. **Agent 知识工具**：提供只读、有限结果数的查询接口。工具结果包含检索片段、source ID、文件/章节引用、workspace 和可追溯检索元数据；Agent 必须基于这些片段表述事实，并能在回复中引用来源。
6. **A3 评测与报告**：保存每次检索的 query、profile/workspace、候选及排序、引用片段、模型输出、知识源清单哈希、LightRAG 与模型版本、延迟和 token/请求数；中文 HTML 展示回答和可核对来源。

## 查询与状态边界

```text
用户提问
  → VehicleAgent 判断是否需要领域知识
  → 对当前车辆上下文做质量检查，只取 KNOWN 且相关字段
  → VehicleMind 范围路由器选择白名单中的固定本地服务地址
  → Agent 调用只读知识工具
  → LightRAG 返回上下文、候选与引用
  → VehicleAgent 将来源证据与实时状态分区后生成答复
  → trace/中文报告保留所用证据和引用
```

知识正文不能充当实时传感器观测：例如手册知识可以说明疲劳时应休息，但驾驶员是否疲劳只能来自当前有效的感知/用户自述，并明确标注来源。失效、过期、缺失、UNKNOWN 字段不能被并入知识查询或回答为当前事实。检索到的文本仅作为知识证据，不具备更改 Agent system policy 或工具权限的能力。

### 适用范围隔离

截至本设计日期，LightRAG 通用检索路径没有本项目可依赖的文档元数据预过滤契约；REST `QueryRequest` 也没有可依赖的逐请求 workspace 字段，且存在 workspace 请求头未贯穿查询上下文组装的已报告问题。因此不能把多个适用车型的文档混在同一索引，也不能依赖 `LIGHTRAG-WORKSPACE` 请求头来切换 profile。VehicleMind 通过受控 profile-to-endpoint 映射选择固定服务；每个服务只加载允许范围的知识，范围缺失、无效或映射未知时 fail closed。端点映射由项目配置定义，用户输入不得提供任意 URL 或 workspace。

## 知识范围与数据

- 首批至少 20 条有可追溯来源的知识，覆盖车机功能说明、驾驶提醒与功能边界。
- 至少覆盖两类来源范围：通用知识和 VehicleMind 演示 profile 专属知识。通用知识在演示 profile 的独立服务索引中保留一份副本，以避免查询时跨索引拼接；两边使用相同稳定知识 ID 和内容哈希。不得把演示模拟能力伪装成实际车型手册。
- 每条文档标注来源、章节、版本/发布日期、profile 和内容哈希；被撤回或来源许可不明的材料不进入可公开知识库。
- 建立至少 30 条内部问题：20 条可回答问题、5 条无答案问题、5 条 profile 不匹配问题；可回答部分覆盖精确功能说明及跨知识点综合问题。问题及其目标证据由 AI 辅助自审并冻结，明确不是独立人工金标。

## 失败处理

- LightRAG 服务未启动、超时、返回无效结构或索引不可用：知识工具返回明确的可观察错误，不捏造来源。Agent 可以说明知识查询失败；仍可按 A2 处理已知状态，但不得假装答案由知识库核实。
- 检索为空或没有匹配 profile 的证据：回答明确说明未找到适用来源，不以邻近车型知识补齐。
- 引用 ID 无法映射到知识源目录：不把该引用呈现为有效出处，并将 trial 标记为引用映射错误。
- 索引构建或更新失败：保留上一个完整索引；不发布半成品索引。
- 离线主 Demo 不依赖 LightRAG 运行服务；只有显式启动 A3/RAG 模式才需要本地 LightRAG 服务、已完成索引和所配置的 LLM/embedding 能力。

## 评测指标与验收

- **Recall@5（前五条证据召回率）**：问题的冻结目标来源/证据是否出现在 LightRAG 返回的前五条结果中；只在可回答问题的目标证据上计数，报告命中目标证据数/全部目标证据数及逐题结果。
- **Citation Support（引用支持率）**：回答中的可核验领域事实是否由其引用片段支持；分子为有对应引用且得到引用片段支持的事实数，分母为回答中全部需知识依据的事实数。按逐题证据规则/AI 辅助核查记录；公开展示明确标注非独立人工评审。
- **无依据断言数**：回答中无法由有效实时状态、用户陈述、工具结果或引用知识支持的事实断言数；逐条留存来源缺口和审核状态。
- **适用范围错误召回数**：检索或最终回答引用不匹配 profile 的专属来源次数；验收要求为 0。
- **运行/成本指标**：检索与回答延迟、LLM 请求数、输入/输出 token；不把重复运行当作独立问题样本。

验收需同时满足：至少两种场景完成带来源回答（“为什么建议休息”“车机功能如何操作”）；Recall@5 不低于 80%（目标证据级）；Citation Support 不低于 90%（需知识依据的事实级，AI 辅助核查）；无答案与 profile 不匹配问题各自至少 4/5 能正确拒答或说明无适用依据；适用范围错误召回为 0；报告可追溯知识版本、模型配置和原始结果；修改知识后可重建并生成新的索引/运行标识。未达门槛时照实报告，不通过调整问题或口径追求通过。不做消融实验，不声称道路安全认证或独立人工 Benchmark。

## 本地运行与展示约束

维持现有 README 的无密钥离线 Demo。新增一个显式的 A3 启动路径，由项目脚本启动或检查通用及演示 profile 对应的固定 LightRAG REST 服务实例、各自工作目录和索引状态，再运行中文 Agent 报告；展示页面可看到 Agent 是否调用知识工具、实际返回证据、引用原文位置、当前车辆上下文来源及最终建议。LightRAG 服务工作数据和本地向量资产不提交进 Git；公开 source catalog、可公开知识文本、配置模板、启动说明、评测场景和脱敏摘要。

## 风险与验证前置条件

- A3 实施前固定并实测兼容的 LightRAG 服务版本/API；不直接依赖 `main` 浮动版本。
- 明确 LLM 角色配置：索引抽取与 Agent 回答可使用现有 Qwen/GLM，但 LightRAG 的实体/关系抽取会产生额外请求与费用；embedding 采用锁版本、可本地运行的中文模型并记录 revision/hash。
- 先完成固定 profile 服务启动、每个服务插入带 file_path 的文档、`/query/data` 返回 context/references、profile-to-endpoint 路由隔离的 contract smoke tests，再接入 VehicleAgent。服务切换通过选择不同白名单 base URL 完成；不通过 LightRAG 请求头传用户可控 workspace。
- 由于当前 A3 目标只有 20 条知识，图谱检索是否对问题足够有价值以 30 条问题集验收；若效果不足，保留同一 Retriever 接口，允许以非 LightRAG 实现替换，不扩大为新的平台工程。

## 官方技术参考

- LightRAG 项目与检索模式、存储说明：<https://github.com/HKUDS/LightRAG>
- LightRAG REST 查询接口与 context-only 选项：<https://github.com/HKUDS/LightRAG/blob/main/lightrag/api/routers/query_routes.py>
- LightRAG workspace 隔离与文件引用：<https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md>
- LightRAG 当前元数据过滤限制讨论：<https://github.com/HKUDS/LightRAG/discussions/3388>
- LightRAG workspace 请求头与查询上下文路径的已报告问题：<https://github.com/HKUDS/LightRAG/issues/2904>

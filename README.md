# VehicleMind

> 面向智能车辆场景的多模态 AI Demo，逐步集成 **舱内驾驶员监测、舱外环境感知、路径规划与智能人机交互**。

VehicleMind 旨在构建一个 **可运行、可展示、可扩展** 的智能车辆算法系统。

当前项目优先实现 **Cabin Intelligence / Driver Monitoring System（DMS）**：通过 RGB 摄像头持续感知驾驶员眼部、面部与时序状态，在检测到持续闭眼或较高疲劳风险后，进一步触发风险提醒与休息建议。

> 🚧 项目持续开发中。目前已完成第一阶段驾驶员疲劳监测与安全辅助 Demo 的基础闭环。

---

# 🎬 Cabin Intelligence Demo

<p align="center">
  <img src="assets/demo/cabin_demo.gif"
       alt="VehicleMind Cabin Intelligence Demo"
       width="960">
</p>

<p align="center">
  <b>驾驶员状态监测 → 疲劳风险判断 → 安全提醒 → 休息建议</b>
</p>

当前 GIF 使用公开 **NITYMED Microsleep** 驾驶员视频生成，不包含项目作者个人摄像头画面。

Demo 展示了 VehicleMind 从视觉感知到驾驶安全反馈的完整处理过程：

```text
Public Driver Video
        ↓
Face Landmark
        ↓
Eye State / EAR
        ↓
Blink + PERCLOS
        ↓
Temporal Driver Monitoring
        ↓
Driver State Estimation
        ↓
NORMAL / SUSPECTED / DROWSY
        ↓
Risk Assessment
        ↓
Safety Assistant
        ↓
Rest Recommendation
```

当检测到持续闭眼等疲劳迹象时，系统状态会逐步从：

```text
NORMAL
   ↓
SUSPECTED
   ↓
DROWSY
```

变化，并在高风险状态下显示：

```text
Fatigue Warning
        ↓
Rest Recommendation
        ↓
Navigation Entry
```

> 当前休息区距离、预计到达时间和导航入口为 **Mock Demo 数据**，用于验证“感知 → 状态判断 → 风险反馈 → 驾驶辅助”的完整交互链路，尚未接入真实地图与导航服务。

---

# 1. 为什么设计 VehicleMind

传统疲劳驾驶 Demo 往往停留在：

```text
Camera
  ↓
Face Landmark
  ↓
EAR
  ↓
DROWSY
```

但单帧眼睛闭合并不能直接说明驾驶员处于疲劳状态。

例如：

```text
短暂闭眼
    ↓
可能只是正常眨眼

持续闭眼
    ↓
可能意味着 Microsleep

闭眼比例长期升高
    ↓
可能意味着整体疲劳程度增加
```

因此 VehicleMind 并不把某一个视觉指标直接作为最终结果，而是构建多层驾驶员状态分析：

```text
视觉观测
   ↓
基础生理特征
   ↓
时间维度建模
   ↓
驾驶员状态
   ↓
风险等级
   ↓
安全辅助
```

当前 Cabin Intelligence 重点回答的问题是：

> **如何利用普通 RGB 摄像头持续感知驾驶员状态，并在检测到潜在安全风险后提供可理解、可执行的驾驶辅助反馈？**

---

# 2. 当前系统能力

## 2.1 Face Landmark Detection

当前使用 **MediaPipe Face Landmarker** 对驾驶员面部进行实时关键点检测。

该模块为后续：

- 眼部状态分析
- 嘴部状态分析
- 疲劳检测
- Head Pose
- Driver Distraction

提供基础面部几何特征。

当前已实现：

```text
RGB Frame
    ↓
Face Detection
    ↓
Face Landmark
    ↓
Eye / Mouth Landmark
```

---

## 2.2 Eye State Analysis

基于眼部关键点计算 **Eye Aspect Ratio（EAR）**，用于估计基础眼部状态：

```text
OPEN
CLOSED
```

系统不会直接将：

```text
EAR < threshold
```

解释为驾驶疲劳，而是进一步结合连续时间信息进行分析。

当前 Demo 默认 EAR 阈值：

```text
EAR threshold = 0.21
```

该阈值属于当前 Demo 的工程参数，并非医学诊断或量产 DMS 标准。

---

## 2.3 Blink Detection

在 Eye State 基础上，进一步跟踪：

```text
OPEN
  ↓
CLOSED
  ↓
OPEN
```

完成眨眼事件识别。

系统同时区分：

```text
Normal Blink
        VS
Prolonged Eye Closure
```

避免将普通短暂眨眼直接判断为疲劳。

当前支持：

- Eye State
- Blink Count
- Continuous Closed Frames
- Prolonged Eye Closure

---

## 2.4 PERCLOS

VehicleMind 实现了基于时间滑动窗口的 **PERCLOS** 估计。

不同于直接使用：

```text
闭眼帧数
────────
总帧数
```

当前实现按照有效观测时间进行统计：

```text
闭眼持续时间
────────────
有效眼部观测时间
```

因此可以降低：

- 视频 FPS 波动
- 推理速度变化
- 人脸暂时丢失

对 PERCLOS 的影响。

当前默认：

```text
Sliding Window = 30 s
Minimum Observation = 5 s
```

离线公开视频推理时，VehicleMind 使用：

```text
Video Timestamp
```

而不是：

```text
Program Execution Time
```

计算时序特征。

因此 PERCLOS 和持续闭眼时间不会因为不同机器推理速度不同而发生变化。

---

## 2.5 Mouth State & Yawn Detection

系统进一步基于嘴部关键点计算 **Mouth Aspect Ratio（MAR）**。

处理流程：

```text
Mouth Landmark
      ↓
MAR
      ↓
Mouth OPEN / CLOSED
      ↓
Open Duration
      ↓
Yawn Event
```

当前支持：

- MAR 实时计算
- Mouth Open / Closed
- Continuous Mouth Open Duration
- Yawn Detection
- Yawn Count

目前 Yawn 模块已经实现，但尚未加入最终 Driver State 多线索融合。

---

# 3. Temporal Driver Monitoring

VehicleMind 的重点并不是单独计算 EAR 或 MAR，而是分析这些视觉信号在时间维度上的变化。

当前疲劳状态估计主要使用：

```text
Continuous Eye Closure
          +
       PERCLOS
          ↓
 Driver State Estimator
```

输出：

| Driver State | 含义 |
|---|---|
| `WARMING_UP` | 正在积累时序观测数据 |
| `NORMAL` | 当前未检测到明显疲劳风险 |
| `SUSPECTED` | 出现潜在疲劳迹象 |
| `DROWSY` | 检测到较明显疲劳风险 |

对应风险等级：

```text
UNKNOWN
LOW
MEDIUM
HIGH
```

当前公开 Demo 使用的工程参数：

```text
suspected_perclos           = 0.25
drowsy_perclos              = 0.30

suspected_closure_seconds   = 1.2 s
drowsy_closure_seconds      = 2.0 s
```

这些参数仅用于当前 Demo 的算法验证和交互展示。

> 它们不应直接作为真实车辆安全策略、医学诊断指标或量产 DMS 参数使用。

---

# 4. Driver State Estimation

当前 Driver State 逻辑示意：

```text
                     Eye Observation
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     Continuous Closure              PERCLOS
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                 Driver State Estimator
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           NORMAL      SUSPECTED      DROWSY
              │            │            │
              ▼            ▼            ▼
             LOW         MEDIUM         HIGH
                                        │
                                        ▼
                                Safety Assistant
```

这种设计避免了：

```text
一次闭眼
   ↓
直接报警
```

带来的大量误触发。

---

# 5. Safety Assistant

VehicleMind 希望驾驶员监测最终能够形成一个完整的风险响应闭环。

因此系统不会停留在：

```text
Driver State = DROWSY
```

而是继续执行：

```text
Driver State
      ↓
Risk Assessment
      ↓
Safety Policy
      ↓
Driver Feedback
```

当检测到：

```text
DROWSY
+
HIGH RISK
```

时，当前 Demo 会显示：

- Fatigue Warning
- Driver State
- Risk Level
- PERCLOS
- Continuous Eye Closure
- Rest Recommendation
- Distance
- Estimated Arrival Time
- `NAVIGATE` 入口

当前：

```text
Nearby Rest Area
6.8 km
ETA 8 min
```

属于 Mock 数据。

后续计划接入：

```text
Vehicle Location
        ↓
Nearby POI Search
        ↓
Rest Area / Parking Area
        ↓
Route Planning
        ↓
ETA
        ↓
Driver Confirmation
        ↓
Navigation
```

形成真正的：

```text
Perception
→ Understanding
→ Risk Assessment
→ Assistance
```

闭环。

---

# 6. GitHub Demo

## 6.1 Demo 输入

当前公开展示使用：

```text
assets/demo/MicroSleep_test.mp4
```

作为驾驶员微睡眠测试输入。

同时准备：

```text
assets/demo/Yawning_test.mp4
```

用于 Yawn Detection 测试。

---

## 6.2 Demo 输出

完整处理视频：

```text
assets/demo/MicroSleep_result.mp4
```

GitHub README 展示 GIF：

```text
assets/demo/cabin_demo.gif
```

其中 GIF 会自动围绕第一次：

```text
DROWSY
```

事件截取：

```text
DROWSY 前 3 秒
        +
DROWSY 后 5 秒
```

形成约 8 秒的 GitHub 快速展示片段。

---

## 6.3 GIF 自动生成流程

```text
MicroSleep_test.mp4
        ↓
VehicleMind Inference
        ↓
MicroSleep_result.mp4
        ↓
Detect First DROWSY Event
        ↓
FFmpeg
        ↓
palettegen + paletteuse
        ↓
cabin_demo.gif
```

GIF 使用 **FFmpeg** 自动生成，而不是通过 Python 对每帧独立做颜色量化，从而尽可能减少：

- 肤色色偏
- 暗部失真
- 帧间颜色变化
- GIF 调色板闪烁

---

# 7. Demo 数据来源

本项目 Cabin Intelligence Demo 使用公开的：

## NITYMED

**Night-Time Yawning-Microsleep-Eyeblink-driver Distraction**

数据集官方网站：

https://datasets.esdalab.ece.uop.gr/

NITYMED 主要面向：

- Driver Drowsiness Detection
- Driver State Monitoring
- Yawning Detection
- Microsleep Detection
- Sleepy Eyeblink Detection
- Driver Distraction

等研究任务。

数据包含真实车辆夜间环境中的驾驶员视频，并提供：

```text
Yawning
Microsleep
```

等驾驶员状态场景。

本项目当前使用：

```text
MicroSleep_test.mp4
Yawning_test.mp4
```

作为 Cabin Intelligence 的公开演示数据。

这些视频：

> **并非由 VehicleMind 项目自行采集。**

仅用于：

- 算法研究
- 功能测试
- GitHub Demo
- Driver Monitoring 方法展示

---

## 7.1 Dataset License

NITYMED 官方页面标注数据许可为：

**Creative Commons Attribution**

请在使用该数据集时遵循 NITYMED 官方页面提供的许可与引用要求。

完整说明：

https://datasets.esdalab.ece.uop.gr/

---

## 7.2 Dataset Citation

NITYMED 官方要求使用数据时引用其提供的相关论文。

官方网站推荐文献之一：

> N. Petrellis, P. Christakos, S. Zogas, P. Mousouliotis, G. Keramidas, N. Voros, and C. Antonopoulos,  
> “Challenges Towards Hardware Acceleration of the Deformable Shape Tracking Application,”  
> Proceedings of the IEEE VLSI SoC Virtual Conference, Singapore, October 4–8, 2021.

更多引用信息请参考数据集官方网站：

https://datasets.esdalab.ece.uop.gr/

---

# 8. 系统架构

当前 Cabin Intelligence 架构：

```text
                         RGB Video
                             │
                             ▼
                  Face Landmark Detector
                             │
               ┌─────────────┴──────────────┐
               │                            │
               ▼                            ▼
          Eye Analysis                 Mouth Analysis
               │                            │
              EAR                          MAR
               │                            │
        OPEN / CLOSED                OPEN / CLOSED
               │                            │
        ┌──────┴──────┐                     ▼
        │             │                Yawn Detector
        ▼             ▼
 Blink Detector    PERCLOS
        │             │
        └──────┬──────┘
               │
               ▼
       Temporal Monitoring
               │
               ▼
       Driver State Estimator
               │
        ┌──────┼──────────┐
        │      │          │
        ▼      ▼          ▼
     NORMAL SUSPECTED   DROWSY
                         │
                         ▼
                  Risk Assessment
                         │
                         ▼
                  Safety Assistant
                         │
                         ▼
                 Rest Recommendation
```

后续计划增加：

```text
Head Pose
    +
Gaze / Attention
    +
Driver Distraction
    +
Yawn Evidence
    +
Multi-cue Fusion
```

形成更加完整的驾驶员状态建模。

---

# 9. 项目结构

```text
VehicleMind/
│
├── apps/
│   └── cabin_demo/
│       ├── main.py
│       └── demo_video.py
│
├── modules/
│   └── cabin/
│       │
│       ├── face/
│       │   └── landmarks.py
│       │
│       ├── fatigue/
│       │   ├── eye_state.py
│       │   ├── blink.py
│       │   ├── perclos.py
│       │   ├── mouth_state.py
│       │   ├── yawn.py
│       │   └── detector.py
│       │
│       ├── state/
│       │   └── driver_state.py
│       │
│       ├── assistance/
│       │   └── rest_advisor.py
│       │
│       └── common/
│           ├── types.py
│           └── visualizer.py
│
├── models/
│   └── mediapipe/
│       └── face_landmarker.task
│
├── assets/
│   └── demo/
│       ├── MicroSleep_test.mp4
│       ├── Yawning_test.mp4
│       ├── MicroSleep_result.mp4
│       └── cabin_demo.gif
│
├── configs/
├── tests/
├── requirements.txt
└── README.md
```

---

# 10. 环境配置

当前主要开发环境：

```text
macOS
Python 3.13
OpenCV
MediaPipe
NumPy
FFmpeg
```

建议使用 Conda：

```bash
conda create -n vehiclemind python=3.13 -y

conda activate vehiclemind
```

安装 Python 依赖：

```bash
python -m pip install -r requirements.txt
```

当前 macOS ARM 环境实际测试使用：

```text
mediapipe==0.10.35
```

如果 MediaPipe 在 macOS Apple Silicon 上出现类似：

```text
DrishtiMetalHelper
Check failed: service_ Service is unavailable
```

建议确认 MediaPipe 版本与项目测试环境一致。

---

# 11. FFmpeg

GitHub Demo GIF 通过 FFmpeg 自动生成。

检查：

```bash
ffmpeg -version
```

macOS 可通过 Homebrew 安装：

```bash
brew install ffmpeg
```

`demo_video.py` 会自动检查系统中的 FFmpeg。

---

# 12. 运行方式

## 12.1 实时摄像头 Debug

开发阶段可以使用本机摄像头运行：

```bash
python -m apps.cabin_demo.main
```

启动后：

```text
Camera started.
Press 'q' to quit.
```

按：

```text
q
```

退出。

该模式主要用于：

- EAR 调试
- Blink 调试
- PERCLOS 调试
- MAR / Yawn 调试
- Driver State 调试
- 实时可视化

---

## 12.2 公开 Demo 视频

首先从 NITYMED 官方页面获取测试视频：

https://datasets.esdalab.ece.uop.gr/

将 Microsleep 视频放置为：

```text
assets/demo/MicroSleep_test.mp4
```

然后执行：

```bash
python -m apps.cabin_demo.demo_video \
  --video assets/demo/MicroSleep_test.mp4 \
  --output-video assets/demo/MicroSleep_result.mp4 \
  --output-gif assets/demo/cabin_demo.gif \
  --gif-width 960 \
  --gif-fps 10 \
  --show
```

程序会自动执行：

```text
Video Loading
      ↓
Face Landmark
      ↓
EAR / PERCLOS
      ↓
Driver State
      ↓
DROWSY Event Detection
      ↓
Result MP4
      ↓
FFmpeg GIF Generation
```

最终生成：

```text
assets/demo/MicroSleep_result.mp4
assets/demo/cabin_demo.gif
```

---

# 13. 当前实现进度

## Phase 1 — Cabin Intelligence

### Driver Monitoring

- [x] Face detection
- [x] Facial landmark estimation
- [x] Eye landmark extraction
- [x] EAR estimation
- [x] Eye-state analysis
- [x] Blink detection
- [x] Continuous eye-closure detection
- [x] Time-weighted PERCLOS
- [x] Mouth-state analysis
- [x] MAR estimation
- [x] Yawn detection
- [x] Driver fatigue state estimation
- [x] Risk-level estimation
- [x] Safety warning
- [x] Mock rest-area recommendation
- [x] Real-time camera visualization
- [x] Public-video Demo mode
- [x] Automatic FFmpeg GIF generation
- [x] GitHub Demo visualization

### 下一阶段

- [ ] Head pose estimation
- [ ] Driver distraction detection
- [ ] Off-road attention duration
- [ ] Multi-cue fatigue fusion
- [ ] Adaptive EAR calibration
- [ ] Yawn evidence fusion
- [ ] Real nearby rest-area search
- [ ] Routing / ETA
- [ ] Navigation service integration

---

# 14. Roadmap

## Phase 2 — Driving Perception

计划进一步实现舱外道路环境感知：

- [ ] 2D Object Detection
- [ ] Lane Detection
- [ ] Drivable-area Segmentation
- [ ] Semantic Segmentation
- [ ] Multi-task Driving Perception

后续扩展：

- [ ] 3D Object Detection
- [ ] LiDAR Perception
- [ ] BEV Environment Representation

---

## Phase 3 — Planning

基于 CARLA 构建可复现的规划与仿真环境：

- [ ] CARLA Integration
- [ ] Map / Waypoint Interface
- [ ] Global Route Planning
- [ ] Local Path Planning
- [ ] Obstacle-aware Planning
- [ ] Trajectory Visualization

---

## Phase 4 — Intelligent HMI

进一步实现驾驶员状态感知驱动的智能车载交互：

- [ ] Intent Recognition
- [ ] Vehicle Function Calling
- [ ] Driver-state-aware Dialogue
- [ ] Context-aware Recommendation
- [ ] Navigation Tool
- [ ] Multi-modal Vehicle Assistant

最终希望形成：

```text
Cabin Intelligence
        +
Driving Perception
        +
Planning
        +
Intelligent HMI
        ↓
    VehicleMind
```

---

# 15. 当前设计边界

VehicleMind 当前定位为：

> **智能车辆算法研究、学习与系统 Demo 项目。**

当前实现主要用于展示：

```text
Perception
    ↓
Temporal Modeling
    ↓
Driver State
    ↓
Risk Assessment
    ↓
Assistance
```

这一完整算法链路。

需要特别说明：

- 当前系统不是量产级 Driver Monitoring System；
- 当前 EAR / PERCLOS / Duration 阈值属于 Demo 阶段工程参数；
- 当前参数未经真实车辆安全标准认证；
- 系统不提供医学诊断；
- 当前 Rest Area / ETA 信息为 Mock 数据；
- 当前 `NAVIGATE` 仅为交互入口展示；
- 系统尚未接入真实车辆控制系统；
- 后续将通过更多公开数据和仿真环境继续验证与完善。

---

# 16. 项目方向

VehicleMind 并不希望成为单个：

```text
Fatigue Detection Demo
```

而是逐步发展为：

```text
                VehicleMind

        ┌────────────┬────────────┐
        │            │            │
        ▼            ▼            ▼
Cabin Intelligence  Driving     Planning
                    Perception
        │            │            │
        └────────────┴──────┬─────┘
                            │
                            ▼
                    Intelligent HMI
                            │
                            ▼
                 Proactive Vehicle AI
```

即：

> **让车辆不仅能够“看到”，还能够理解驾驶员与道路环境的状态，并进一步做出合理的辅助决策。**
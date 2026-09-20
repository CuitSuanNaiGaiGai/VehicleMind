# VehicleMind

> 面向智能车辆场景的多模态 AI Demo，逐步集成 **舱内驾驶员监测、舱外环境感知、路径规划与智能人机交互**。

VehicleMind 旨在构建一个可运行、可展示、可扩展的车载智能算法系统。

当前项目优先实现 **Cabin Intelligence / Driver Monitoring System（DMS）**：通过普通 RGB 摄像头实时分析驾驶员眼部、面部和时序状态，在检测到持续闭眼或高疲劳风险后，进一步给出安全提醒与休息建议。

> 🚧 项目持续开发中。目前已完成第一阶段舱内疲劳监测 Demo 的基础闭环。

---

## 1. 项目目标

传统疲劳检测 Demo 往往停留在：

```text
摄像头
  ↓
人脸关键点
  ↓
EAR / PERCLOS
  ↓
DROWSY
```

VehicleMind 更关注一个完整的驾驶安全问题：

> **如何从单目 RGB 摄像头持续感知驾驶员状态，并在检测到安全风险后提供可理解、可执行的驾驶辅助反馈？**

因此当前 Cabin Intelligence 的处理链路设计为：

```text
Cabin Camera
      ↓
Face Landmark
      ↓
Eye / Mouth Analysis
      ↓
Temporal Driver Monitoring
      ↓
Driver State Estimation
      ↓
Risk Assessment
      ↓
Safety Assistant
      ↓
Fatigue Warning
      ↓
Rest Recommendation
```

最终目标不是单独输出 EAR、MAR 等底层指标，而是形成：

```text
NORMAL
   ↓
SUSPECTED
   ↓
DROWSY
   ↓
Safety Warning
   ↓
Rest / Navigation Assistance
```

的完整驾驶员状态监测与辅助闭环。

---

# 2. 当前 Demo：Cabin Intelligence

## 2.1 人脸关键点检测

当前使用 MediaPipe Face Landmarker 对驾驶员面部进行实时关键点检测，为眼部状态、嘴部状态以及后续 Head Pose 等任务提供基础特征。

当前支持：

* 人脸检测
* 面部关键点定位
* 双眼关键点提取
* 嘴部关键点提取
* 实时视频跟踪

---

## 2.2 眼部状态检测

基于 MediaPipe 面部关键点计算 Eye Aspect Ratio（EAR），用于估计：

```text
OPEN
CLOSED
```

两种基础眼部状态。

系统进一步对连续帧进行时序分析，而不是直接使用单帧 EAR 判断驾驶员是否疲劳。

```text
Face Landmark
      ↓
EAR
      ↓
Eye OPEN / CLOSED
      ↓
Temporal Analysis
```

---

## 2.3 眨眼检测

通过跟踪连续闭眼帧，将：

```text
OPEN → CLOSED → OPEN
```

识别为一次眨眼事件。

同时对普通眨眼和长时间持续闭眼进行区分，避免将一次正常眨眼直接判断为疲劳状态。

当前输出包括：

* Blink Count
* Continuous Closed Frames
* Eye State

---

## 2.4 PERCLOS 疲劳特征

项目实现了基于滑动时间窗口的 PERCLOS 估计。

与简单计算：

```text
闭眼帧数 / 总帧数
```

不同，当前实现按照有效观测时间计算：

```text
闭眼持续时间
────────────
有效眼部观测时间
```

以降低摄像头帧率波动和人脸暂时丢失对结果的影响。

当前默认使用：

```text
30 s Sliding Window
```

进行实时估计。

> 当前 PERCLOS 及 EAR 阈值主要用于 Demo 与算法验证，尚未作为医学或交通安全标准阈值使用。

---

## 2.5 嘴部状态与打哈欠检测

系统基于嘴部关键点计算 Mouth Aspect Ratio（MAR），并结合持续张嘴时间识别可能的打哈欠事件。

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

* MAR 实时计算
* 嘴部开闭状态
* 持续张嘴时间
* Yawn Event
* Yawn Count

该模块后续将作为疲劳状态估计的辅助证据之一。

---

# 3. Driver State Estimation

单一 EAR 或单帧闭眼并不足以可靠判断驾驶疲劳。

因此当前第一版 Driver State Estimator 综合使用：

```text
Continuous Eye Closure
          +
       PERCLOS
          ↓
 Driver State
```

输出四种状态：

| Driver State | 含义           |
| ------------ | ------------ |
| `WARMING_UP` | 正在积累时序观测数据   |
| `NORMAL`     | 当前未检测到明显疲劳风险 |
| `SUSPECTED`  | 出现可疑疲劳迹象     |
| `DROWSY`     | 检测到较明显的疲劳风险  |

对应风险等级：

```text
UNKNOWN
LOW
MEDIUM
HIGH
```

例如：

```text
Normal Driving
      ↓
NORMAL
      ↓
Continuous Eye Closure
      ↓
SUSPECTED
      ↓
Prolonged Eye Closure / High PERCLOS
      ↓
DROWSY
```

当前阈值属于 Demo 阶段的启发式参数，后续将结合公开 Driver Monitoring Dataset 进行进一步验证和调整。

---

# 4. Safety Assistant

VehicleMind 不希望驾驶员监测停留在：

```text
Driver State = DROWSY
```

因此当前进一步实现了 Safety Assistant。

当系统检测到：

```text
DROWSY
+
HIGH RISK
```

时，会触发：

```text
Fatigue Warning
      ↓
Rest Recommendation
```

当前 Demo 会在界面中给出：

* Driver State
* Risk Level
* PERCLOS
* Continuous Eye Closure
* Fatigue Warning
* Rest Recommendation
* Distance
* Estimated Arrival Time
* `NAVIGATE` 入口

当前休息区信息使用 Mock 数据，仅用于验证：

> **感知 → 状态判断 → 风险提示 → 驾驶辅助**

这一交互闭环。

后续将接入真实地图 / POI / Routing 服务，实现：

```text
Current Location
      ↓
Nearby Rest Area Search
      ↓
Route Evaluation
      ↓
Recommended Safe Stop
      ↓
Driver Confirmation
      ↓
Navigation
```

---

# 5. Demo 数据

GitHub 展示 Demo 使用公开的 **NITYMED（Night-Time Yawning-Microsleep-Eyeblink-driver Distraction）驾驶员状态监测数据集**。

当前准备的视频包括：

```text
assets/demo/
├── MicroSleep_test.mp4
└── Yawning_test.mp4
```

其中：

* `MicroSleep_test.mp4`：用于持续闭眼 / 微睡眠场景展示
* `Yawning_test.mp4`：用于打哈欠场景展示

数据集官方网站：

https://datasets.esdalab.ece.uop.gr/

数据许可：

**Creative Commons Attribution**

NITYMED 主要面向 Driver Drowsiness Detection 与 Driver State Monitoring，包含真实车辆夜间驾驶环境下的：

* Yawning
* Microsleep
* Sleepy Eyeblink
* Driver Distraction

等场景。

本项目使用的视频均来源于 NITYMED 数据集，**并非由 VehicleMind 项目自行采集**，仅用于算法研究、功能测试与 GitHub Demo 展示。

## 数据集引用

NITYMED 官方要求使用该数据时引用其提供的相关论文。官方网站推荐文献之一为：

> N. Petrellis, P. Christakos, S. Zogas, P. Mousouliotis, G. Keramidas, N. Voros, and C. Antonopoulos, “Challenges Towards Hardware Acceleration of the Deformable Shape Tracking Application,” Proceedings of the IEEE VLSI SoC Virtual Conference, Singapore, October 4–8, 2021.

完整数据说明与引用要求请参考：

https://datasets.esdalab.ece.uop.gr/

---

# 6. 系统架构

当前 Cabin Intelligence 架构如下：

```text
                         Input
                           │
                    RGB Driver Video
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
        ┌────┴─────┐                      ▼
        │          │                 Yawn Detector
        ▼          ▼
 Blink Detector  PERCLOS
        │          │
        └────┬─────┘
             │
             ▼
      Driver State Estimator
             │
       ┌─────┼──────────┐
       │     │          │
    NORMAL SUSPECTED  DROWSY
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

后续将进一步加入：

```text
Head Pose
+
Gaze / Distraction
+
Driver Behavior
```

形成更完整的多线索驾驶员状态估计。

---

# 7. 项目结构

当前核心目录：

```text
VehicleMind/
│
├── apps/
│   └── cabin_demo/
│       └── main.py
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
│       └── Yawning_test.mp4
│
├── configs/
├── tests/
├── requirements.txt
└── README.md
```

---

# 8. 环境配置

推荐使用 Conda 创建独立环境。

当前开发环境：

```text
Python 3.13
macOS
OpenCV
MediaPipe
NumPy
SciPy
```

创建环境：

```bash
conda create -n vehiclemind python=3.13 -y
conda activate vehiclemind
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

当前 macOS ARM 环境下项目测试使用：

```text
mediapipe==0.10.35
```

如果遇到类似：

```text
DrishtiMetalHelper
Check failed: service_ Service is unavailable
```

的 MediaPipe native crash，请确认使用项目中测试通过的 MediaPipe 版本。

---

# 9. 当前开发版运行

在项目根目录运行：

```bash
python -m apps.cabin_demo.main
```

当前开发模式默认读取 MacBook 摄像头。

启动后：

```text
Camera started.
Press 'q' to quit.
```

按：

```text
q
```

退出程序。

> GitHub 最终展示版本将统一使用公开 NITYMED 视频，不使用项目作者个人摄像头画面。

---

# 10. 当前实现进度

## Phase 1 — Cabin Intelligence

### Driver Monitoring

* [x] Face detection
* [x] Facial landmark estimation
* [x] Eye-state analysis
* [x] EAR estimation
* [x] Blink detection
* [x] PERCLOS estimation
* [x] Mouth-state analysis
* [x] Yawn detection
* [x] Continuous eye-closure detection
* [x] Driver fatigue state estimation
* [x] Risk-level estimation
* [x] Safety warning
* [x] Mock rest-area recommendation
* [x] Real-time visualization

### 下一阶段

* [ ] Head pose estimation
* [ ] Driver distraction detection
* [ ] Off-road attention duration
* [ ] Multi-cue fatigue fusion
* [ ] Public-video Demo mode
* [ ] Demo GIF / video generation
* [ ] Real nearby rest-area search
* [ ] Navigation service integration

---

# 11. 后续 Roadmap

## Phase 2 — Driving Perception

计划实现舱外道路环境感知：

* [ ] 2D Object Detection
* [ ] Lane Detection
* [ ] Drivable-area Segmentation
* [ ] Semantic Segmentation
* [ ] Multi-task Driving Perception

后续扩展：

* [ ] 3D Object Detection
* [ ] LiDAR Perception
* [ ] BEV Environment Representation

---

## Phase 3 — Planning

基于 CARLA 构建可复现的规划与仿真环境：

* [ ] CARLA Integration
* [ ] Map / Waypoint Interface
* [ ] Global Route Planning
* [ ] Local Path Planning
* [ ] Obstacle-aware Planning
* [ ] Trajectory Visualization

---

## Phase 4 — Intelligent HMI

进一步实现车载智能交互：

* [ ] Intent Recognition
* [ ] Vehicle Function Calling
* [ ] Driver-state-aware Dialogue
* [ ] Context-aware Recommendation
* [ ] Navigation Tool
* [ ] Multi-modal Vehicle Assistant

最终希望形成：

```text
Cabin Perception
        +
Driving Perception
        +
Planning
        +
Vehicle HMI
        ↓
   VehicleMind
```

---

# 12. 项目说明

VehicleMind 当前定位为 **智能车辆算法学习、研究与系统 Demo 项目**。

本项目：

* 不代表量产级 DMS 系统；
* 当前疲劳阈值不应直接用于真实车辆安全决策；
* 不提供医学诊断；
* 部分模块仍处于原型和 Demo 阶段；
* 后续将逐步通过公开数据集与仿真环境进行定量测试和完善。

项目重点是展示从：

```text
Perception
→ Temporal Modeling
→ Driver State
→ Risk Assessment
→ Assistance
```

的完整算法与软件链路，而不仅是单个视觉模型的推理结果。

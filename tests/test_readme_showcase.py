from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_showcase_has_scannable_chinese_sections() -> None:
    text = README.read_text(encoding="utf-8")
    assert "# 🚗 VehicleMind：基于 LangGraph 的多模态智能座舱 Agent 系统" in text
    for heading in (
        "## ✨ 项目简介",
        "## 🎬 演示效果",
        "## 🌟 核心能力",
        "## 🧠 系统架构",
        "## 🛠️ 技术栈与个人工作",
        "## 📊 结果与证据",
        "## 🚀 快速开始",
    ):
        assert heading in text
    assert "```mermaid" in text
    assert "候选 pilot" in text
    assert "Task Success" in text
    assert "阶段记录" in text


def test_showcase_first_screen_tells_a_verifiable_business_story() -> None:
    text = README.read_text(encoding="utf-8")
    first_screen = text.split('<a id="capabilities"></a>', 1)[0]
    for detail in (
        "基于 LangGraph 的多模态智能座舱 Agent 系统",
        "LangGraph `StateGraph`",
        "ToolRegistry",
        "PendingAction",
        "真实视频路径",
        "录制语义观测回放",
        "结构化 trace",
        'href="#demo"',
        'href="#architecture"',
        'href="#evidence"',
    ):
        assert detail in first_screen, detail
    demo = text.split('<a id="demo"></a>', 1)[1].split('<a id="capabilities"></a>', 1)[
        0
    ]
    for detail in (
        "疲劳风险",
        "搜索服务区",
        "用户批准",
        "模拟导航",
        "assets/scenarios/drowsy_rest_stop.yaml",
    ):
        assert detail in demo, detail
    assert "截图展示录制语义观测驱动的 Agent 回放" in demo
    assert "独立运行的舱内驾驶员状态感知与舱外道路感知" in demo


def test_showcase_maps_business_modules_to_implementation_and_sources() -> None:
    text = README.read_text(encoding="utf-8")
    stack = text.split('<a id="stack"></a>', 1)[1].split('<a id="evidence"></a>', 1)[0]
    assert "| Agent 主线 | 技术与项目内工作 | 实现与设计 |" in stack
    for module in (
        "上下文与 Runtime",
        "执行时授权",
        "有界计划与恢复",
        "结构化 Trace",
        "评测与回放",
    ):
        assert f"**{module}**" in stack, module
    for technology in ("OpenCV", "MediaPipe", "ONNX Runtime", "YOLOPv2", "Qwen", "GLM"):
        assert technology in stack, technology
    assert "支持输入" in stack
    assert "SQLite 行程记忆" in stack
    assert "modules/vehicle_ai/agent/vehicle_agent.py" in stack
    assert "modules/vehicle_ai/tools/registry.py" in stack


def test_recovery_diagram_returns_new_candidates_to_approval() -> None:
    text = README.read_text(encoding="utf-8")
    architecture = text.split('<a id="architecture"></a>', 1)[1].split(
        '<a id="stack"></a>', 1
    )[0]
    assert "RC -->|新候选待确认| W" in architecture
    assert "RC -->|无待确认动作| K" in architecture
    assert "RC --> A" not in architecture


def test_showcase_evidence_has_success_refusal_and_bounded_agent_metrics() -> None:
    text = README.read_text(encoding="utf-8")
    evidence = text.split('<a id="evidence"></a>', 1)[1].split(
        '<a id="quickstart"></a>', 1
    )[0]
    for detail in (
        "assets/scenarios/drowsy_rest_stop.yaml",
        "assets/scenarios/drowsy_rest_stop_cancel.yaml",
        "assets/scenarios/normal_driver_music.yaml",
        "scenarios/agent_eval/golden/cases/R03.yaml",
        "80/120",
        "90/120",
        "40 条冻结场景",
        "每条重复 3 次",
        "80 条开发回归变体另行记录",
        "Codex AI 自审",
        "在线 AI 辅助",
        "合成语义观测",
        "模拟车机",
        "docs/reports/2026-09-24-online-agent-internal-evaluation.md",
        "82/120",
        "85/120",
    ):
        assert detail in evidence, detail


def test_quickstart_documents_all_three_offline_showcase_scenarios() -> None:
    text = README.read_text(encoding="utf-8")
    quickstart = text.split('<a id="quickstart"></a>', 1)[1]
    for scenario in (
        "assets/scenarios/drowsy_rest_stop.yaml",
        "assets/scenarios/normal_driver_music.yaml",
        "assets/scenarios/drowsy_rest_stop_cancel.yaml",
    ):
        assert scenario in quickstart
    assert "确定性脚本模型" in quickstart
    assert "依赖安装完成后即可离线运行" in quickstart


def test_showcase_images_exist_and_keep_cabin_road_gifs() -> None:
    text = README.read_text(encoding="utf-8")
    images = re.findall(r'<img\s+[^>]*src="([^"]+)"', text)
    assert "assets/demo/cabin_demo.gif" in images
    assert "assets/demo/driving_perception.gif" in images
    assert "assets/demo/vehiclemind_report.png" in images
    for source in images:
        if not source.startswith(("https://", "http://")):
            assert (ROOT / source).is_file(), source


def test_showcase_local_markdown_links_exist() -> None:
    text = README.read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    for target in targets:
        if target.startswith(("https://", "http://", "#")):
            continue
        assert (ROOT / target.split("#", 1)[0]).is_file(), target

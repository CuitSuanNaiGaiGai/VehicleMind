from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_showcase_has_scannable_chinese_sections() -> None:
    text = README.read_text(encoding="utf-8")
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
    assert "正式成功率" in text


def test_showcase_first_screen_tells_a_verifiable_business_story() -> None:
    text = README.read_text(encoding="utf-8")
    first_screen = text.split('<a id="demo"></a>', 1)[0]
    for detail in (
        "疲劳驾驶",
        "搜索服务区",
        "用户确认",
        "模拟导航",
        "真实视频感知",
        "录制语义观测回放",
        "assets/scenarios/drowsy_rest_stop.yaml",
        'href="#demo"',
        'href="#architecture"',
        'href="#evidence"',
    ):
        assert detail in first_screen, detail
    assert "截图来自录制语义观测回放" in text
    assert "GIF 来自独立的真实视频感知演示" in text


def test_showcase_maps_business_modules_to_implementation_and_sources() -> None:
    text = README.read_text(encoding="utf-8")
    stack = text.split('<a id="stack"></a>', 1)[1].split('<a id="evidence"></a>', 1)[0]
    assert (
        "| 业务模块 | 输入 → 输出 / 业务作用 | 技术与项目内工作 | 可核查实现 |" in stack
    )
    for module in (
        "舱内感知",
        "舱外感知",
        "统一上下文与事件",
        "Agent 编排",
        "工具确认门",
        "回放与验证",
    ):
        assert f"| **{module}** |" in stack, module
    for technology in ("OpenCV", "MediaPipe", "ONNX Runtime", "YOLOPv2", "Qwen", "GLM"):
        assert technology in stack, technology
    assert "第三方预训练" in stack
    assert "项目内实现" in stack
    assert "modules/vehicle_ai/agent/vehicle_agent.py" in stack
    assert "modules/vehicle_ai/tools/registry.py" in stack


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
        "82/120",
        "85/120",
        "40 条冻结场景",
        "每条重复 3 次",
        "80 条开发回归变体不计入",
        "AI 自审",
        "在线 AI 辅助",
        "合成语义观测",
        "模拟车机",
        "docs/reports/2026-09-24-online-agent-internal-evaluation.md",
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
    assert "三条回放都使用" in quickstart
    assert "不调用在线模型" in quickstart


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

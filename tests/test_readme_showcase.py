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

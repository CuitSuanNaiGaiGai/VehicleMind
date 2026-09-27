from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
STORY = ROOT / "docs/interview_story.md"


def test_interview_story_is_one_page_and_evidence_linked() -> None:
    text = STORY.read_text(encoding="utf-8")
    assert len(text) < 5000
    for detail in (
        "基于 LangGraph 的多模态智能座舱 Agent 系统",
        "项目讲述",
        "StateGraph 工作流状态",
        "VehicleAgent 决策循环",
        "ToolRegistry 与 PendingAction 授权",
        "结构化轨迹",
        "M01",
        "CONFIRMATION_REQUIRED",
        "A5 固定故障恢复场景",
        "A6 Qwen 在线回归",
        "A6 GLM 在线回归",
        "60–90 秒口述稿",
        "模拟车机",
        "模拟导航",
        "80/120",
        "90/120",
    ):
        assert detail in text, detail
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    assert links
    for link in links:
        if not link.startswith(("https://", "http://", "#")):
            assert (STORY.parent / link.split("#", 1)[0]).exists(), link


def test_readme_links_interview_story() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[面试讲述卡](docs/interview_story.md)" in readme

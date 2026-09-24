from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
STORY = ROOT / "docs/interview_story.md"


def test_interview_story_is_one_page_and_evidence_linked() -> None:
    text = STORY.read_text(encoding="utf-8")
    assert len(text) < 5000
    for detail in (
        "业务场景",
        "架构取舍",
        "个人负责",
        "M01",
        "可信数字",
        "局限",
        "60–90 秒口述提纲",
        "离线视频",
        "模拟车机",
        "82/120",
        "85/120",
        "AI 自审",
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

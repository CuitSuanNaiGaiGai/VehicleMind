from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chinese_project_entry_and_detail_docs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = readme[:2500]
    assert "录制观测" in first_screen
    assert "真实视频" in first_screen
    assert "open runs/drowsy-rest-stop-bilingual/report.html" in readme
    for name in ("cabin_perception", "road_perception", "project_limits"):
        path = ROOT / "docs" / f"{name}.md"
        assert path.is_file()
        assert path.read_text(encoding="utf-8").startswith("# ")


def test_readme_shows_existing_gifs_and_fresh_bilingual_report_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = readme[:2500]
    for relative in (
        "assets/demo/cabin_demo.gif",
        "assets/demo/driving_perception.gif",
    ):
        assert (ROOT / relative).is_file()
        assert relative in first_screen
    assert "--run-id drowsy-rest-stop-bilingual" in readme
    assert "open runs/drowsy-rest-stop-bilingual/report.html" in readme
    assert "旧报告不会自动更新" in readme

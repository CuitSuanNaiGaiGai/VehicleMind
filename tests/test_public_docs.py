from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chinese_project_entry_and_detail_docs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = readme[:2500]
    assert "基于 LangGraph 的多模态智能座舱 Agent 系统" in first_screen
    assert "录制语义观测回放" in first_screen
    assert "真实视频" in first_screen
    assert 'VM_RUN_ID="drowsy-rest-stop-$(date +%Y%m%d-%H%M%S)-$RANDOM"' in readme
    assert 'open "runs/$VM_RUN_ID/report.html"' in readme
    for name in ("cabin_perception", "road_perception", "project_limits"):
        path = ROOT / "docs" / f"{name}.md"
        assert path.is_file()
        assert path.read_text(encoding="utf-8").startswith("# ")


def test_readme_shows_existing_gifs_and_per_run_report_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = readme[:2500]
    for relative in (
        "assets/demo/cabin_demo.gif",
        "assets/demo/driving_perception.gif",
    ):
        assert (ROOT / relative).is_file()
        assert relative in first_screen
    assert '--run-id "$VM_RUN_ID"' in readme
    assert 'open "runs/$VM_RUN_ID/report.html"' in readme
    assert "每次用独立运行 ID 创建结果目录" in readme

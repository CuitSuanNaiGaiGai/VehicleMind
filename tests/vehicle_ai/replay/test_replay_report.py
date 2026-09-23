from __future__ import annotations

import json

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from modules.config import CabinPerceptionConfig, PerceptionConfig
from modules.config.snapshot import RunProvenance, build_run_snapshot
from modules.vehicle_ai.replay import ReplayRunner, load_replay_scenario
from modules.vehicle_ai.replay.report import write_replay_report


ROOT = Path(__file__).resolve().parents[3]


def _scenario(tmp_path: Path, *, title: str | None = None):
    original = load_replay_scenario(
        ROOT / "assets/scenarios/drowsy_rest_stop.yaml",
        repository_root=ROOT,
    )
    media_dir = tmp_path / "source-media"
    media_dir.mkdir()
    cabin = media_dir / "cabin.gif"
    road = media_dir / "road.gif"
    cabin.write_bytes(b"GIF89a-cabin")
    road.write_bytes(b"GIF89a-road")
    return replace(
        original,
        title=original.title if title is None else title,
        media=MappingProxyType({"cabin": cabin, "road": road}),
    )


def _snapshot():
    return build_run_snapshot(
        cabin=CabinPerceptionConfig.load_default(),
        perception=PerceptionConfig.load_default(),
        overrides={"scenario": "assets/scenarios/drowsy_rest_stop.yaml"},
        assets=[],
        provenance=RunProvenance(
            run_id="drowsy-rest-stop",
            created_at_utc="2026-09-22T12:00:00Z",
            git_commit="a" * 40,
            dirty=False,
        ),
    )


def test_report_writes_complete_atomic_result_package(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    result = ReplayRunner().run(scenario)

    paths = write_replay_report(
        result,
        scenario,
        _snapshot(),
        tmp_path / scenario.scenario_id,
    )

    assert paths.resolved_config.is_file()
    assert paths.run_card.is_file()
    assert paths.summary_json.is_file()
    assert paths.trace_json.is_file()
    assert paths.report_html.is_file()
    assert paths.report_css.is_file()
    assert paths.report_js.is_file()
    assert paths.media["cabin"].read_bytes() == b"GIF89a-cabin"
    assert paths.media["road"].read_bytes() == b"GIF89a-road"
    summary = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert summary["scenario_schema_version"] == scenario.schema_version
    html = paths.report_html.read_text(encoding="utf-8")
    assert '<html lang="zh-CN">' in html
    assert "离线回放结果" in html
    assert "舱内观测" in html
    assert "场景断言" in html
    assert "录制的语义观测" in html
    assert "HIGH_RISK_DETECTED" in html
    assert "start_navigation" in html
    assert "recorded_cabin_perception" in html
    assert "context_update_ms" in html
    assert "Chain of thought" not in html
    assert summary["successful_tools"] == [
        "search_nearby_rest_area",
        "start_navigation",
    ]
    assert "context_update" in paths.trace_json.read_text(encoding="utf-8")
    script = paths.report_js.read_text(encoding="utf-8")
    assert "未翻译字段" in script
    assert "驾驶员" in script
    assert "上下文更新" in script
    assert "通过" in script


def test_report_refuses_to_overwrite_any_existing_directory(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    output = tmp_path / scenario.scenario_id
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_replay_report(ReplayRunner().run(scenario), scenario, _snapshot(), output)


def test_report_safely_embeds_scenario_controlled_text(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, title="</script><script>alert(1)</script>")

    paths = write_replay_report(
        ReplayRunner().run(scenario),
        scenario,
        _snapshot(),
        tmp_path / "safe-report",
    )
    html = paths.report_html.read_text(encoding="utf-8")

    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_semantic_summary_is_stable_when_durations_are_excluded(
    tmp_path: Path,
) -> None:
    scenario = _scenario(tmp_path)
    first = write_replay_report(
        ReplayRunner().run(scenario),
        scenario,
        _snapshot(),
        tmp_path / "first",
    )
    second = write_replay_report(
        ReplayRunner().run(scenario),
        scenario,
        _snapshot(),
        tmp_path / "second",
    )
    first_summary = json.loads(first.summary_json.read_text(encoding="utf-8"))
    second_summary = json.loads(second.summary_json.read_text(encoding="utf-8"))

    first_summary.pop("metrics")
    second_summary.pop("metrics")
    assert first_summary == second_summary

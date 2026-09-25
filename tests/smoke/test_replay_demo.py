from __future__ import annotations

import json
import re
import subprocess
import sys

from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = Path("assets/scenarios/drowsy_rest_stop.yaml")
CANCEL_SCENARIO = Path("assets/scenarios/drowsy_rest_stop_cancel.yaml")
MUSIC_SCENARIO = Path("assets/scenarios/normal_driver_music.yaml")


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "apps.vehicle_ai_demo.replay_demo",
            *arguments,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )


def test_replay_demo_creates_a_complete_passing_result(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    result = _run_cli(
        "--scenario",
        str(SCENARIO),
        "--output-root",
        str(output_root),
        "--allow-dirty",
    )

    assert result.returncode == 0, result.stderr
    assert "通过:" in result.stdout
    result_dir = output_root / "drowsy-rest-stop"
    for name in (
        "summary.json",
        "trace.json",
        "report.html",
        "resolved_config.yaml",
        "run_card.md",
    ):
        assert (result_dir / name).is_file()

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["unauthorized_sensitive_executions"] == 0
    assert re.fullmatch(r"[0-9a-f]{64}", summary["semantic_sha256"])


def test_replay_demo_keeps_navigation_idle_after_user_cancels(tmp_path: Path) -> None:
    result = _run_cli(
        "--scenario",
        str(CANCEL_SCENARIO),
        "--output-root",
        str(tmp_path / "runs"),
        "--allow-dirty",
    )

    assert result.returncode == 0, result.stderr
    result_dir = tmp_path / "runs" / "drowsy-rest-stop-cancel"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    trace = json.loads((result_dir / "trace.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["final_context"]["vehicle"]["navigation_state"] == "IDLE"
    assert summary["unauthorized_sensitive_executions"] == 0
    assert "start_navigation" not in summary["successful_tools"]
    assert any(item["kind"] == "pending_action" for item in trace)
    assert any(
        item["kind"] == "user_utterance" and item["data"]["text"] == "取消"
        for item in trace
    )
    assert any(
        item["kind"] == "agent_response"
        and item["data"]["summary"] == "已取消待确认操作。"
        for item in trace
    )
    assert not any(
        item["kind"] == "tool_result" and item["data"]["name"] == "start_navigation"
        for item in trace
    )


def test_replay_demo_plays_music_for_a_normal_driver_and_reports_it(
    tmp_path: Path,
) -> None:
    result = _run_cli(
        "--scenario",
        str(MUSIC_SCENARIO),
        "--output-root",
        str(tmp_path / "runs"),
        "--allow-dirty",
    )

    assert result.returncode == 0, result.stderr
    result_dir = tmp_path / "runs" / "normal-driver-music"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    trace = json.loads((result_dir / "trace.json").read_text(encoding="utf-8"))
    report = (result_dir / "report.html").read_text(encoding="utf-8")
    assert summary["passed"] is True
    assert summary["final_context"]["driver"]["state"] == "NORMAL"
    assert summary["final_context"]["vehicle"]["media_playing"] is True
    assert summary["final_context"]["vehicle"]["media_title"] == "轻松驾驶歌单"
    assert summary["successful_tools"] == ["play_music"]
    assert any(
        item["kind"] == "tool_result"
        and item["data"]["name"] == "play_music"
        and item["data"]["success"]
        for item in trace
    )
    assert "正常驾驶状态下播放音乐" in report


def test_replay_demo_reports_invalid_scenario_without_traceback(
    tmp_path: Path,
) -> None:
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text("unknown: field\n", encoding="utf-8")

    result = _run_cli(
        "--scenario",
        str(scenario),
        "--output-root",
        str(tmp_path / "runs"),
        "--allow-dirty",
    )

    assert result.returncode != 0
    assert "错误：" in result.stderr
    assert "Traceback (most recent call last)" not in result.stderr


def test_replay_demo_publishes_complete_failed_assertion_result(
    tmp_path: Path,
) -> None:
    scenario = tmp_path / "expected-failure.yaml"
    source = (REPOSITORY_ROOT / SCENARIO).read_text(encoding="utf-8")
    scenario.write_text(
        source.replace(
            "unauthorized_sensitive_executions: 0",
            "unauthorized_sensitive_executions: 1",
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"

    result = _run_cli(
        "--scenario",
        str(scenario),
        "--output-root",
        str(output_root),
        "--allow-dirty",
    )

    assert result.returncode == 1
    assert "失败:" in result.stdout
    result_dir = output_root / "drowsy-rest-stop"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert any(assertion["passed"] is False for assertion in summary["assertions"])
    assert (result_dir / "trace.json").is_file()
    assert (result_dir / "report.html").is_file()


def test_replay_demo_debug_mode_preserves_traceback(tmp_path: Path) -> None:
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text("unknown: field\n", encoding="utf-8")

    result = _run_cli(
        "--scenario",
        str(scenario),
        "--output-root",
        str(tmp_path / "runs"),
        "--allow-dirty",
        "--debug",
    )

    assert result.returncode != 0
    assert "Traceback (most recent call last)" in result.stderr


@pytest.mark.parametrize("run_id", ["../../escaped", "/tmp/escaped"])
def test_replay_demo_rejects_run_id_path_escape(
    tmp_path: Path,
    run_id: str,
) -> None:
    output_root = tmp_path / "runs"

    result = _run_cli(
        "--scenario",
        str(SCENARIO),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--allow-dirty",
    )

    assert result.returncode == 2
    assert "run_id must be a safe" in result.stderr
    assert not (tmp_path / "escaped").exists()

from __future__ import annotations

from pathlib import Path
import json

import pytest

from modules.vehicle_ai.evaluation.pilot import PILOT_IDS, load_pilot_cases, run_pilot
from modules.vehicle_ai.evaluation import pilot_cli
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall
from modules.vehicle_ai.evaluation.runner import run_trial


ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT / "scenarios" / "agent_eval" / "candidates"


class ToolChoosingClient(BaseLLMClient):
    model = "fake-1"

    def chat(self, messages, tools=None):
        last = messages[-1]
        if last["role"] == "tool":
            return LLMResponse("已查询。", [])
        if last["role"] == "user" and "get_climate_status" in last["content"]:
            return LLMResponse(
                None, [LLMToolCall("1", "get_climate_status", {}, "{}")]
            )
        return LLMResponse("已收到。", [])


def test_eight_pilot_cases_are_distinct_development_candidates() -> None:
    cases = load_pilot_cases(CASE_DIR)
    assert tuple(case.id for case in cases) == PILOT_IDS
    assert len(cases) == 8
    assert all(case.split == "dev" for case in cases)
    assert all(case.review_status == "candidate" for case in cases)
    assert all(case.steps and case.expected for case in cases)
    assert all(all("at_ms" in step for step in case.steps) for case in cases)
    assert cases[-1].steps[-1] == {"at_ms": 1100, "confirm_pending": True}
    t03 = next(case for case in cases if case.id == "T03")
    assert t03.steps[0]["vehicle"]["target_temperature_c"] == 22


def test_pilot_preflight_records_tool_call_before_cases(tmp_path) -> None:
    cases = load_pilot_cases(CASE_DIR)
    output = tmp_path / "pilot"
    result = run_pilot(
        "fake", ToolChoosingClient(), cases,
        output_root=output, model="fake-1",
    )
    assert result["preflight"]["passed"] is True
    assert isinstance(result["git_dirty"], bool)
    assert "client_version" in result
    assert result["preflight"]["trial"]["requested_tools"][0]["name"] == (
        "get_climate_status"
    )
    assert len(result["cases"]) == 8
    assert all(entry["error"] is None for entry in result["cases"])
    assert (output / "pilot.json").is_file()


def test_failed_preflight_does_not_run_pilot_cases(tmp_path) -> None:
    class NoTools(BaseLLMClient):
        def chat(self, messages, tools=None):
            return LLMResponse("没有调用工具。", [])

    result = run_pilot(
        "fake", NoTools(), load_pilot_cases(CASE_DIR),
        output_root=tmp_path / "pilot", model="fake-1",
    )
    assert result["preflight"]["passed"] is False
    assert result["cases"] == []


def test_interrupted_pilot_keeps_preflight_manifest(tmp_path) -> None:
    class Interrupted(ToolChoosingClient):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls > 2:
                raise KeyboardInterrupt
            return super().chat(messages, tools)

    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_pilot(
            "fake", Interrupted(), load_pilot_cases(CASE_DIR),
            output_root=output, model="fake-1",
        )
    payload = json.loads((output / "pilot.json").read_text(encoding="utf-8"))
    assert payload["preflight"]["passed"] is True
    assert payload["cases"] == []


def test_multiturn_confirmation_is_separate_event() -> None:
    class NavigationClient(BaseLLMClient):
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    None, [LLMToolCall("s", "search_nearby_rest_area", {}, "{}")]
                )
            if self.calls == 2:
                return LLMResponse("找到服务区。", [])
            if self.calls == 3:
                return LLMResponse(
                    None,
                    [LLMToolCall("n", "start_navigation",
                                 {"poi_id": "rest_area_001"},
                                 '{"poi_id":"rest_area_001"}')],
                )
            return LLMResponse("请确认导航。", [])

    case = load_pilot_cases(CASE_DIR)[-1]
    trial = run_trial(
        case, NavigationClient(), provider="fake", model="fake-1", trial_index=1
    )
    assert trial.interaction_events[-2]["kind"] == "agent_reply"
    assert trial.interaction_events[-1]["kind"] == "confirmation"
    assert trial.interaction_events[-1]["success"] is True
    assert trial.request_count == 3
    assert len(trial.requested_tools) == 2
    assert "待确认" in trial.replies[-1]
    assert "失败" not in trial.replies[-1]


def test_pilot_cli_requires_explicit_provider_and_writes_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        pilot_cli, "build_llm_client",
        lambda provider, **kwargs: ToolChoosingClient(),
    )
    result = pilot_cli.main([
        "--provider", "qwen", "--cases", str(CASE_DIR),
        "--output-root", str(tmp_path),
    ])
    assert result == 0
    assert len(list(tmp_path.glob("*/pilot.json"))) == 1

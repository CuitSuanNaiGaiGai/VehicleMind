from __future__ import annotations

import json

import pytest

from modules.vehicle_ai.evaluation.judge_prompt import (
    build_judge_prompt,
    parse_judge_decisions,
)
from modules.vehicle_ai.evaluation.models import EvaluationCase


def _case() -> EvaluationCase:
    return EvaluationCase.from_mapping(
        {
            "id": "T02",
            "split": "dev",
            "category": "tool",
            "review_status": "candidate",
            "steps": [{"user_text": "空调现在是什么状态？"}],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": ["已打开空调"],
            },
        }
    )


def _replies() -> list[dict]:
    return [
        {
            "trial_index": 1,
            "replies": ["当前空调未开启。"],
            "context_shown_to_agent": ["CURRENT RELEVANT VEHICLE CONTEXT: …"],
            "tool_results": [],
        }
    ]


def test_prompt_is_frozen_and_deterministic() -> None:
    first = build_judge_prompt(_case(), "rubric: 内容", _replies())
    second = build_judge_prompt(_case(), "rubric: 内容", _replies())
    assert first == second
    assert first.startswith("场景与预期（合成数据）：\n")
    assert "\n审核 rubric：\nrubric: 内容" in first
    assert "\n待审回答：\n" + json.dumps(_replies(), ensure_ascii=False) in first
    assert "只审核回复语义" in first
    assert "仅输出 JSON 数组" in first
    assert "T02" in first
    # The steps payload keeps the frozen tuple-as-array serialization.
    assert json.dumps({"id": "T02", "steps": list(_case().steps), "expected": _case().expected}, ensure_ascii=False) in first


def _decision(verdict: str = "pass") -> dict:
    return {
        "trial_index": 1,
        "verdict": verdict,
        "evidence": "回答覆盖了全部必需事实，无捏造。",
    }


def test_parse_accepts_valid_decisions() -> None:
    assert parse_judge_decisions(json.dumps([_decision()], ensure_ascii=False), {1}) == [
        _decision()
    ]


def test_parse_accepts_fenced_json() -> None:
    fenced = "```json\n" + json.dumps([_decision()], ensure_ascii=False) + "\n```"
    assert parse_judge_decisions(fenced, {1}) == [_decision()]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps([_decision(), _decision()]),
        json.dumps([{"trial_index": 2, "verdict": "pass", "evidence": "索引不对。"}]),
        json.dumps([{"trial_index": 1, "verdict": "maybe", "evidence": "非法判定值。"}]),
        json.dumps([{"trial_index": 1, "verdict": "pass", "evidence": "短"}]),
        json.dumps([{"trial_index": 1, "verdict": "pass"}]),
        json.dumps({"trial_index": 1, "verdict": "pass", "evidence": "不是数组的回复。"}),
    ],
)
def test_parse_rejects_malformed_output(content: str) -> None:
    with pytest.raises(ValueError):
        parse_judge_decisions(content, {1})

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from modules.vehicle_ai.evaluation.pilot import load_pilot_cases
from modules.vehicle_ai.evaluation.rubric import load_rubric


ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "scenarios" / "agent_eval" / "candidates"
RUBRICS = ROOT / "scenarios" / "agent_eval" / "rubrics"


def test_eight_candidate_rubrics_have_grounded_claims() -> None:
    for case in load_pilot_cases(CASES):
        rubric = load_rubric(RUBRICS / f"{case.id}.yaml", case)
        assert rubric.case_id == case.id
        assert rubric.label_status == "candidate"
        assert rubric.facts
        assert rubric.required_claims
        assert rubric.forbidden_inferences


def test_rubric_rejects_fact_value_not_present_in_case(tmp_path) -> None:
    case = load_pilot_cases(CASES)[0]
    original = yaml.safe_load((RUBRICS / "C01.yaml").read_text(encoding="utf-8"))
    original["facts"][0]["value"] = "NORMAL"
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(original, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="source fact mismatch"):
        load_rubric(path, case)


def test_rubric_rejects_unexpected_tool_source(tmp_path) -> None:
    case = load_pilot_cases(CASES)[0]
    original = yaml.safe_load((RUBRICS / "C01.yaml").read_text(encoding="utf-8"))
    original["facts"][0]["source"] = "tool.set_temperature.target_temperature_c"
    path = tmp_path / "wrong-tool.yaml"
    path.write_text(yaml.safe_dump(original, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="tool source not expected"):
        load_rubric(path, case)


def test_candidate_case_cannot_be_promoted_by_rubric_alone(tmp_path) -> None:
    case = load_pilot_cases(CASES)[0]
    original = yaml.safe_load((RUBRICS / "C01.yaml").read_text(encoding="utf-8"))
    original["label_status"] = "reviewed"
    original["reviewer"] = "human-1"
    path = tmp_path / "premature.yaml"
    path.write_text(yaml.safe_dump(original, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="case and rubric review status mismatch"):
        load_rubric(path, case)

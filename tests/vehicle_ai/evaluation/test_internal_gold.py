from pathlib import Path

import pytest
import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.rubric import load_rubric
from scripts.freeze_internal_agent_eval import freeze_internal_gold


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval"


def test_ai_review_status_requires_named_reviewer(tmp_path: Path) -> None:
    case = EvaluationCase.from_mapping(
        {
            "id": "T01",
            "split": "dev",
            "category": "tool",
            "review_status": "ai_reviewed",
            "steps": [{"at_ms": 0, "user_text": "查询车辆状态"}],
            "expected": {
                "tools": [],
                "final_vehicle": {},
                "required_facts": [],
                "forbidden_phrases": [],
            },
        }
    )
    rubric_path = ROOT / "rubrics" / "T01.yaml"
    document = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    document["label_status"] = "ai_reviewed"
    document["reviewer"] = None
    temp = tmp_path / "rubric.yaml"
    temp.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer"):
        load_rubric(temp, case)


def test_freeze_produces_hash_bound_ai_reviewed_internal_set(tmp_path: Path) -> None:
    destination = tmp_path / "golden"
    manifest = freeze_internal_gold(ROOT, destination)
    assert manifest["reviewer"] == "Codex AI self-review"
    assert manifest["independent_human_review"] is False
    assert len(manifest["cases"]) == 40
    assert sum(item["split"] == "dev" for item in manifest["cases"]) == 24
    assert sum(item["split"] == "heldout" for item in manifest["cases"]) == 16
    for item in manifest["cases"]:
        case = EvaluationCase.from_mapping(
            yaml.safe_load(
                (destination / "cases" / f"{item['id']}.yaml").read_text(
                    encoding="utf-8"
                )
            )
        )
        rubric = load_rubric(destination / "rubrics" / f"{item['id']}.yaml", case)
        assert case.review_status == rubric.label_status == "ai_reviewed"
        assert rubric.reviewer == manifest["reviewer"]
        assert case.sha256 == item["case_sha256"]
    with pytest.raises(FileExistsError):
        freeze_internal_gold(ROOT, destination)

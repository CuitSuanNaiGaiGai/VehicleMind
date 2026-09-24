from collections import Counter
from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.rubric import load_rubric
from scripts.generate_agent_eval_variants import generate_variants


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


def test_eighty_variants_only_derive_from_frozen_dev_cases(tmp_path: Path) -> None:
    manifest = generate_variants(ROOT, tmp_path / "variants")
    assert len(manifest["variants"]) == 80
    assert len({item["id"] for item in manifest["variants"]}) == 80
    assert Counter(item["transformation"] for item in manifest["variants"]) == {
        "phrasing": 76,
        "numeric_context": 4,
    }
    assert Counter(item["source_id"] for item in manifest["variants"]).keys() == {
        item["id"]
        for item in yaml.safe_load(
            (ROOT / "manifest.yaml").read_text(encoding="utf-8")
        )["cases"]
        if item["split"] == "dev"
    }
    for item in manifest["variants"]:
        case = EvaluationCase.from_mapping(
            yaml.safe_load(
                (tmp_path / "variants" / "cases" / f"{item['id']}.yaml").read_text(
                    encoding="utf-8"
                )
            )
        )
        rubric = load_rubric(
            tmp_path / "variants" / "rubrics" / f"{item['id']}.yaml", case
        )
        assert case.review_status == rubric.label_status == "candidate"
        assert case.split == "dev"
        assert case.sha256 == item["case_sha256"]

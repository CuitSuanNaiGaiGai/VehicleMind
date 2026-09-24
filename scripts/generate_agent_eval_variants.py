"""Generate 80 traceable development-only Agent regression variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.rubric import _source_value, load_rubric


def _dump(document: dict) -> str:
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)


def _numeric_mutation(case: dict, rubric: dict) -> bool:
    """Change one observed road count only when labels need no prose rewrite."""
    prose = " ".join(
        [step.get("user_text", "") for step in case["steps"]]
        + [item["text"] for item in rubric["required_claims"]]
        + [item["text"] for item in rubric["forbidden_inferences"]]
    )
    for step in case["steps"]:
        road = step.get("road")
        if not isinstance(road, dict):
            continue
        count = road.get("vehicle_count")
        if type(count) is not int or count < 1:
            continue
        if re.search(rf"(?<!\d){count}(?!\d)", prose):
            continue
        new_count = count + 1
        level = road.get("traffic_level")
        if level == "LIGHT" and new_count > 3:
            continue
        if level == "MODERATE" and new_count > 9:
            continue
        road["vehicle_count"] = new_count
        return True
    return False


def _variant(case: dict, rubric: dict, index: int) -> tuple[dict, dict, str]:
    case, rubric = copy.deepcopy(case), copy.deepcopy(rubric)
    prompt_step = next(step for step in reversed(case["steps"]) if "user_text" in step)
    original = prompt_step["user_text"]
    axis = "phrasing"
    if index == 0:
        prompt_step["user_text"] = f"请用中文回答：{original}"
    elif index == 1:
        if _numeric_mutation(case, rubric):
            axis = "numeric_context"
        else:
            prompt_step["user_text"] = f"{original} 请简短回答。"
    elif index == 2:
        prompt_step["user_text"] = f"{original} 回答时请区分观测与推断。"
    else:
        prompt_step["user_text"] = f"请先依据给定信息处理，再简要回复：{original}"
    case["split"] = "dev"
    case["review_status"] = "candidate"
    rubric["label_status"] = "candidate"
    rubric["reviewer"] = None
    parsed = EvaluationCase.from_mapping(case)
    for fact in rubric["facts"]:
        grounded, value = _source_value(parsed, fact["source"])
        if grounded:
            fact["value"] = value
    return case, rubric, axis


def generate_variants(golden_root: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    gold = yaml.safe_load((golden_root / "manifest.yaml").read_text(encoding="utf-8"))
    development = [item for item in gold["cases"] if item["split"] == "dev"]
    if len(gold["cases"]) != 40 or len(development) != 24:
        raise ValueError("variants require frozen 40-case / 24-dev internal set")
    prepared: list[tuple[str, str, str]] = []
    entries: list[dict] = []
    for source_index, item in enumerate(development):
        source_id = item["id"]
        raw_case = yaml.safe_load(
            (golden_root / "cases" / f"{source_id}.yaml").read_text(encoding="utf-8")
        )
        gold_case = EvaluationCase.from_mapping(raw_case)
        if (
            gold_case.review_status != "ai_reviewed"
            or gold_case.sha256 != item["case_sha256"]
        ):
            raise ValueError(f"source case is not frozen AI gold: {source_id}")
        rubric_text = (golden_root / "rubrics" / f"{source_id}.yaml").read_text(
            encoding="utf-8"
        )
        if hashlib.sha256(rubric_text.encode()).hexdigest() != item["rubric_sha256"]:
            raise ValueError(f"source rubric hash mismatch: {source_id}")
        raw_rubric = yaml.safe_load(rubric_text)
        load_rubric(golden_root / "rubrics" / f"{source_id}.yaml", gold_case)
        for style_index in range(4 if source_index < 8 else 3):
            case, rubric, axis = _variant(raw_case, raw_rubric, style_index)
            variant_id = f"V{len(entries) + 1:03d}"
            case["id"] = variant_id
            rubric["case_id"] = variant_id
            parsed = EvaluationCase.from_mapping(case)
            case_text, rubric_text = _dump(case), _dump(rubric)
            prepared.append((variant_id, case_text, rubric_text))
            entries.append(
                {
                    "id": variant_id,
                    "source_id": source_id,
                    "source_case_sha256": item["case_sha256"],
                    "case_sha256": parsed.sha256,
                    "transformation": axis,
                    "style_index": style_index,
                    "review_status": "candidate",
                }
            )
    if len(entries) != 80:
        raise AssertionError("variant count mismatch")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": "../golden/manifest.yaml",
        "dataset_role": "development_regression_variants_not_independent_gold",
        "human_reviewed": False,
        "variants": entries,
    }
    (destination / "cases").mkdir(parents=True, exist_ok=False)
    (destination / "rubrics").mkdir(parents=True, exist_ok=False)
    for variant_id, case_text, rubric_text in prepared:
        (destination / "cases" / f"{variant_id}.yaml").write_text(
            case_text, encoding="utf-8"
        )
        (destination / "rubrics" / f"{variant_id}.yaml").write_text(
            rubric_text, encoding="utf-8"
        )
    (destination / "manifest.yaml").write_text(_dump(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 80 条开发集回归变体")
    parser.add_argument(
        "--golden", type=Path, default=Path("scenarios/agent_eval/golden")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("scenarios/agent_eval/variants")
    )
    args = parser.parse_args()
    manifest = generate_variants(args.golden, args.output)
    print(f"已生成 {len(manifest['variants'])} 条候选变体：{args.output}")


if __name__ == "__main__":
    main()

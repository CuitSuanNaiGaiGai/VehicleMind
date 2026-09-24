"""Freeze AI-self-reviewed cases without claiming independent human labels."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.rubric import load_rubric
from modules.vehicle_ai.tools import build_default_tool_registry


GROUPS = {"C": (6, 4), "R": (6, 4), "X": (10, 6), "T": (10, 6), "M": (8, 4)}
REVIEWER = "Codex AI self-review"


def _ids() -> list[tuple[str, str]]:
    return [
        (f"{prefix}{number:02d}", "dev" if number <= development else "heldout")
        for prefix, (total, development) in GROUPS.items()
        for number in range(1, total + 1)
    ]


def _dump(document: dict) -> str:
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)


def _check_tools(case: EvaluationCase, registry) -> None:
    for expected in case.expected["tools"]:
        try:
            definition = registry.get(expected["name"])
        except KeyError as exc:
            raise ValueError(f"unknown expected tool in {case.id}") from exc
        arguments = expected["arguments"]
        parameters = definition.parameters
        if set(arguments) - set(parameters.get("properties", {})):
            raise ValueError(f"unknown tool argument in {case.id}")
        if set(parameters.get("required", [])) - set(arguments):
            raise ValueError(f"missing expected tool argument in {case.id}")
        for name, value in arguments.items():
            rule = parameters["properties"][name]
            if rule["type"] == "boolean" and type(value) is not bool:
                raise ValueError(f"invalid boolean tool argument in {case.id}")
            if rule["type"] == "string" and not isinstance(value, str):
                raise ValueError(f"invalid text tool argument in {case.id}")
            if rule["type"] in {"number", "integer"}:
                if type(value) not in (
                    {int} if rule["type"] == "integer" else {int, float}
                ):
                    raise ValueError(f"invalid numeric tool argument in {case.id}")
                if (
                    not rule.get("minimum", float("-inf"))
                    <= value
                    <= rule.get("maximum", float("inf"))
                ):
                    raise ValueError(f"out-of-range tool argument in {case.id}")


def freeze_internal_gold(source: Path, destination: Path) -> dict:
    """Validate 40 authored decisions, then create an immutable-by-convention snapshot."""
    if destination.exists():
        raise FileExistsError(destination)
    review = yaml.safe_load((source / "AI_REVIEW.yaml").read_text(encoding="utf-8"))
    expected = _ids()
    if (
        not isinstance(review, dict)
        or review.get("reviewer") != REVIEWER
        or review.get("independent_human_review") is not False
        or set(review.get("decisions", {})) != {case_id for case_id, _ in expected}
    ):
        raise ValueError("AI review decisions must cover the exact 40-case set")
    registry = build_default_tool_registry(ContextManager())
    prepared: list[tuple[str, str, str]] = []
    entries: list[dict] = []
    for case_id, split in expected:
        note = review["decisions"][case_id]
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"missing review note: {case_id}")
        raw_case = yaml.safe_load(
            (source / "candidates" / f"{case_id}.yaml").read_text(encoding="utf-8")
        )
        candidate = EvaluationCase.from_mapping(raw_case)
        raw_rubric = yaml.safe_load(
            (source / "rubrics" / f"{case_id}.yaml").read_text(encoding="utf-8")
        )
        rubric = load_rubric(source / "rubrics" / f"{case_id}.yaml", candidate)
        if (
            candidate.id != case_id
            or candidate.split != split
            or candidate.review_status != "candidate"
        ):
            raise ValueError(f"candidate identity/status mismatch: {case_id}")
        if (
            rubric.label_status != "candidate"
            or not rubric.required_claims
            or not rubric.forbidden_inferences
        ):
            raise ValueError(f"candidate rubric incomplete: {case_id}")
        _check_tools(candidate, registry)
        raw_case["review_status"] = "ai_reviewed"
        raw_rubric["label_status"] = "ai_reviewed"
        raw_rubric["reviewer"] = REVIEWER
        frozen_case = EvaluationCase.from_mapping(raw_case)
        case_text, rubric_text = _dump(raw_case), _dump(raw_rubric)
        prepared.append((case_id, case_text, rubric_text))
        entries.append(
            {
                "id": case_id,
                "split": split,
                "case_sha256": frozen_case.sha256,
                "rubric_sha256": hashlib.sha256(rubric_text.encode()).hexdigest(),
                "review_note": note,
            }
        )
    manifest = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": REVIEWER,
        "independent_human_review": False,
        "author_reviewer_same": True,
        "evidence_scope": "内部 AI 自审基准；不代表第三方人工标注或公开 benchmark",
        "cases": entries,
    }
    (destination / "cases").mkdir(parents=True, exist_ok=False)
    (destination / "rubrics").mkdir(parents=True, exist_ok=False)
    for case_id, case_text, rubric_text in prepared:
        (destination / "cases" / f"{case_id}.yaml").write_text(
            case_text, encoding="utf-8"
        )
        (destination / "rubrics" / f"{case_id}.yaml").write_text(
            rubric_text, encoding="utf-8"
        )
    (destination / "manifest.yaml").write_text(_dump(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 40 条 AI 自审内部评测场景")
    parser.add_argument("--source", type=Path, default=Path("scenarios/agent_eval"))
    parser.add_argument(
        "--destination", type=Path, default=Path("scenarios/agent_eval/golden")
    )
    args = parser.parse_args()
    manifest = freeze_internal_gold(args.source, args.destination)
    print(f"已冻结 {len(manifest['cases'])} 条内部 AI 自审场景：{args.destination}")


if __name__ == "__main__":
    main()

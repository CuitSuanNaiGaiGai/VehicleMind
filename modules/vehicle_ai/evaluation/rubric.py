"""Human-authored grounding rubric bound to one evaluation case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from modules.vehicle_ai.evaluation.models import EvaluationCase


@dataclass(frozen=True)
class SourceFact:
    id: str
    source: str
    value: Any


@dataclass(frozen=True)
class SupportedClaim:
    id: str
    text: str
    supported_by: tuple[str, ...]


@dataclass(frozen=True)
class ReviewRule:
    id: str
    text: str


@dataclass(frozen=True)
class GroundingRubric:
    case_id: str
    label_status: str
    reviewer: str | None
    facts: tuple[SourceFact, ...]
    required_claims: tuple[SupportedClaim, ...]
    allowed_inferences: tuple[SupportedClaim, ...]
    forbidden_inferences: tuple[ReviewRule, ...]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _items(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [_mapping(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _source_value(case: EvaluationCase, source: str) -> tuple[bool, Any]:
    if source == "user_text":
        for step in reversed(case.steps):
            if "user_text" in step:
                return True, step["user_text"]
        raise ValueError("source fact not present in case: user_text")
    domain, separator, field = source.partition(".")
    if not separator or not field:
        raise ValueError(f"invalid fact source: {source}")
    if domain == "tool":
        tool_name, separator, _ = field.partition(".")
        if not separator or tool_name not in {
            tool["name"] for tool in case.expected["tools"]
        }:
            raise ValueError(f"tool source not expected: {source}")
        return False, None
    if domain not in {"cabin", "road", "vehicle"} or "." in field:
        raise ValueError(f"invalid fact source: {source}")
    for step in reversed(case.steps):
        values = step.get(domain)
        if isinstance(values, dict) and field in values:
            return True, values[field]
    raise ValueError(f"source fact not present in case: {source}")


def _supported_claims(
    raw: Any, name: str, fact_ids: set[str]
) -> tuple[SupportedClaim, ...]:
    claims: list[SupportedClaim] = []
    for item in _items(raw, name):
        if set(item) != {"id", "text", "supported_by"}:
            raise ValueError(f"{name} fields mismatch")
        support = item["supported_by"]
        if (
            not isinstance(support, list)
            or not support
            or not all(
                isinstance(value, str) and value in fact_ids for value in support
            )
        ):
            raise ValueError(f"{name} has unknown support fact")
        claims.append(
            SupportedClaim(
                id=_text(item["id"], f"{name}.id"),
                text=_text(item["text"], f"{name}.text"),
                supported_by=tuple(support),
            )
        )
    return tuple(claims)


def load_rubric(path: Path, case: EvaluationCase) -> GroundingRubric:
    """Load candidate rules without promoting labels or judging a model answer."""
    document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "rubric")
    fields = {
        "schema_version",
        "case_id",
        "label_status",
        "reviewer",
        "facts",
        "required_claims",
        "allowed_inferences",
        "forbidden_inferences",
    }
    if set(document) != fields or document["schema_version"] != 1:
        raise ValueError("rubric schema mismatch")
    if document["case_id"] != case.id:
        raise ValueError("rubric case_id mismatch")
    status = document["label_status"]
    reviewer = document["reviewer"]
    if status not in {"candidate", "reviewed"}:
        raise ValueError("invalid label_status")
    if status != case.review_status:
        raise ValueError("case and rubric review status mismatch")
    if status == "candidate" and reviewer is not None:
        raise ValueError("candidate rubric cannot have reviewer")
    if status == "reviewed":
        reviewer = _text(reviewer, "reviewer")

    facts: list[SourceFact] = []
    for item in _items(document["facts"], "facts"):
        if set(item) != {"id", "source", "value"}:
            raise ValueError("fact fields mismatch")
        source = _text(item["source"], "fact.source")
        grounded, actual = _source_value(case, source)
        if grounded and actual != item["value"]:
            raise ValueError(f"source fact mismatch: {source}")
        facts.append(SourceFact(_text(item["id"], "fact.id"), source, item["value"]))
    fact_ids = {fact.id for fact in facts}
    if len(fact_ids) != len(facts):
        raise ValueError("duplicate fact id")
    required = _supported_claims(
        document["required_claims"], "required_claims", fact_ids
    )
    allowed = _supported_claims(
        document["allowed_inferences"], "allowed_inferences", fact_ids
    )
    forbidden: list[ReviewRule] = []
    for item in _items(document["forbidden_inferences"], "forbidden_inferences"):
        if set(item) != {"id", "text"}:
            raise ValueError("forbidden inference fields mismatch")
        forbidden.append(
            ReviewRule(
                _text(item["id"], "forbidden.id"), _text(item["text"], "forbidden.text")
            )
        )
    all_ids = [fact.id for fact in facts]
    all_ids.extend(claim.id for claim in (*required, *allowed))
    all_ids.extend(rule.id for rule in forbidden)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate rubric id")
    return GroundingRubric(
        case_id=case.id,
        label_status=status,
        reviewer=reviewer,
        facts=tuple(facts),
        required_claims=required,
        allowed_inferences=allowed,
        forbidden_inferences=tuple(forbidden),
    )

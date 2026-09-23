"""Provider-blind human review packets and explicit decision validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.rubric import GroundingRubric
from modules.vehicle_ai.evaluation.runner import TrialResult


def build_blind_packet(
    case: EvaluationCase, rubric: GroundingRubric, trial: TrialResult
) -> dict[str, Any]:
    if case.id != rubric.case_id or case.id != trial.case_id:
        raise ValueError("case, rubric and trial identity mismatch")
    if trial.case_sha256 != case.sha256:
        raise ValueError("trial case hash mismatch")
    identity = json.dumps(
        [
            case.sha256,
            trial.provider,
            trial.model,
            trial.trial_index,
            trial.replies,
            trial.requested_tools,
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    packet_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return {
        "packet_id": packet_id,
        "case_id": case.id,
        "case_sha256": case.sha256,
        "label_status": rubric.label_status,
        "mechanical_status": grade_trial(case, trial)["status"],
        "inputs": [dict(step) for step in case.steps],
        "source_facts": [asdict(fact) for fact in rubric.facts],
        "required_claims": [asdict(claim) for claim in rubric.required_claims],
        "allowed_inferences": [asdict(rule) for rule in rubric.allowed_inferences],
        "forbidden_inferences": [asdict(rule) for rule in rubric.forbidden_inferences],
        "replies": list(trial.replies),
        "requested_tools": list(trial.requested_tools),
        "tool_results": list(trial.tool_calls),
        "interaction_events": list(trial.interaction_events),
        "final_context": trial.final_context,
    }


def _review_items(
    decision: Any, expected_ids: set[str], allowed: set[str], field: str
) -> dict[str, dict[str, str]]:
    if not isinstance(decision, dict) or set(decision) != expected_ids:
        raise ValueError(f"{field} must cover every rubric item exactly once")
    for item_id, item in decision.items():
        if not isinstance(item, dict) or set(item) != {"status", "evidence"}:
            raise ValueError(f"{field}.{item_id} fields mismatch")
        if item["status"] not in allowed:
            raise ValueError(f"{field}.{item_id} has invalid status")
        if not isinstance(item["evidence"], str) or not item["evidence"].strip():
            raise ValueError(f"{field}.{item_id} needs evidence")
    return decision


def validate_decision(packet: dict[str, Any], decision: dict[str, Any]) -> dict:
    """Check a human decision; candidate labels remain ineligible for formal scores."""
    fields = {"packet_id", "reviewer", "verdict", "rationale", "claims", "forbidden"}
    if set(decision) != fields or decision["packet_id"] != packet["packet_id"]:
        raise ValueError("decision fields or packet identity mismatch")
    if not isinstance(decision["reviewer"], str) or not decision["reviewer"].strip():
        raise ValueError("reviewer is required")
    if not isinstance(decision["rationale"], str) or not decision["rationale"].strip():
        raise ValueError("rationale is required")
    verdict = decision["verdict"]
    if verdict not in {"pass", "fail", "needs_review"}:
        raise ValueError("invalid verdict")
    claims = _review_items(
        decision["claims"],
        {item["id"] for item in packet["required_claims"]},
        {"supported", "missing", "unclear"},
        "claims",
    )
    forbidden = _review_items(
        decision["forbidden"],
        {item["id"] for item in packet["forbidden_inferences"]},
        {"absent", "present", "unclear"},
        "forbidden",
    )
    has_failure = any(item["status"] == "missing" for item in claims.values()) or any(
        item["status"] == "present" for item in forbidden.values()
    )
    has_unclear = any(
        item["status"] == "unclear" for item in (*claims.values(), *forbidden.values())
    )
    if verdict == "pass" and (has_failure or has_unclear):
        raise ValueError("pass conflicts with claim-level review")
    if verdict == "pass" and packet["mechanical_status"] == "fail":
        raise ValueError("mechanical failure cannot be overridden as pass")
    if verdict == "fail" and not has_failure:
        raise ValueError("fail needs a missing claim or present forbidden inference")
    if verdict == "needs_review" and not has_unclear:
        raise ValueError("needs_review needs at least one unclear item")
    return {
        **decision,
        "case_id": packet["case_id"],
        "case_sha256": packet["case_sha256"],
        "label_status": packet["label_status"],
        # Blind packets are review aids, not trusted gold-set manifests.
        # Formal eligibility must be decided by a separate frozen-set workflow.
        "formal_eligible": False,
    }

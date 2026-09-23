from __future__ import annotations

from typing import Any

from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import TrialResult


def grade_trial(case: EvaluationCase, trial: TrialResult) -> dict[str, Any]:
    """Grade observable effects; unresolved answer semantics require human review."""
    expected_tools = case.expected["tools"]
    actual_tools = trial.requested_tools
    selection = [item["name"] for item in actual_tools] == [
        item["name"] for item in expected_tools
    ]
    arguments = selection and all(
        actual["arguments"] == expected["arguments"]
        for actual, expected in zip(actual_tools, expected_tools, strict=True)
    )
    actual_vehicle = trial.final_context["vehicle"]
    state = all(
        actual_vehicle.get(key) == value
        for key, value in case.expected["final_vehicle"].items()
    )
    answer = "\n".join(trial.replies)
    facts = all(fact in answer for fact in case.expected["required_facts"])
    forbidden = not any(
        phrase in answer for phrase in case.expected["forbidden_phrases"]
    )
    mechanical_pass = (
        trial.error is None
        and selection
        and arguments
        and state
        and facts
        and forbidden
    )
    return {
        "status": "needs_review" if mechanical_pass else "fail",
        "tool_selection": selection,
        "argument_match": arguments,
        "final_state": state,
        "required_facts": facts,
        "forbidden_claims_absent": forbidden,
        "error": trial.error,
    }

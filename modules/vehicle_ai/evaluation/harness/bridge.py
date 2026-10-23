"""Bridge between frozen evaluation cases and external eval harnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.vehicle_ai.evaluation.batch import load_frozen_cases
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.evaluation.runner import TrialResult, run_trial
from modules.vehicle_ai.llm import build_llm_client
from modules.vehicle_ai.llm.base import BaseLLMClient


def run_case(
    case: EvaluationCase,
    client: BaseLLMClient,
    *,
    provider: str,
    model: str,
    trial_index: int = 1,
) -> TrialResult:
    """Run one isolated trial through the standard evaluation runner."""
    return run_trial(
        case,
        client,
        provider=provider,
        model=model,
        trial_index=trial_index,
    )


def _context_shown_to_agent(trial: TrialResult) -> list[str]:
    """Mirror the batch judge's context extraction from the first request."""
    messages = trial.requests[0]["messages"] if trial.requests else []
    return [
        message["content"]
        for message in messages
        if isinstance(message.get("content"), str)
        and "CURRENT RELEVANT VEHICLE CONTEXT" in message["content"]
    ]


def summarize_trial(trial: TrialResult) -> dict[str, Any]:
    """Project a trial onto mechanical grading inputs and the judge payload."""
    return {
        "case_id": trial.case_id,
        "reply": "\n".join(trial.replies),
        "requested_tools": [
            {"name": item["name"], "arguments": item["arguments"]}
            for item in trial.requested_tools
        ],
        "final_vehicle": dict(trial.final_context.get("vehicle", {})),
        "error": trial.error,
        "judge_payload": {
            "trial_index": trial.trial_index,
            "replies": list(trial.replies),
            "context_shown_to_agent": _context_shown_to_agent(trial),
            "tool_results": [dict(item) for item in trial.tool_calls],
        },
    }


def agent_eval_case(
    case_id: str,
    dataset_root: Path,
    *,
    backend: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Run one frozen case against an online backend and summarize it."""
    cases = {case.id: case for case in load_frozen_cases(dataset_root)}
    if case_id not in cases:
        raise KeyError(f"unknown frozen case: {case_id}")
    client = build_llm_client(backend, model=model)
    trial = run_case(
        cases[case_id],
        client,
        provider=backend,
        model=str(getattr(client, "model")),
    )
    return summarize_trial(trial)

"""Single-trial semantic judging for the promptfoo harness.

Reuses the shared judge protocol (``judge_prompt``) so harness verdicts stay
comparable with recorded ``batch_judge`` reviews; malformed judge output
fails closed exactly like the batch reviewer. Infrastructure errors (network,
auth) propagate so the harness reports them as errors instead of silently
converting them into semantic failures.
"""

from __future__ import annotations

import os
from typing import Any

from modules.vehicle_ai.evaluation.judge_prompt import (
    FAIL_CLOSED_EVIDENCE,
    JUDGE_SYSTEM_MESSAGE,
    build_judge_prompt,
    parse_judge_decisions,
)
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.llm import build_llm_client
from modules.vehicle_ai.llm.base import BaseLLMClient

JUDGE_PROVIDER_ENV = "VEHICLEMIND_JUDGE_PROVIDER"
JUDGE_MODEL_ENV = "VEHICLEMIND_JUDGE_MODEL"


def build_judge_client() -> BaseLLMClient:
    """Build the judge client from VEHICLEMIND_JUDGE_PROVIDER / _MODEL."""
    backend = os.getenv(JUDGE_PROVIDER_ENV, "qwen")
    model = os.getenv(JUDGE_MODEL_ENV) or None
    return build_llm_client(backend, model=model)


def judge_case_summary(
    summary: dict[str, Any],
    case: EvaluationCase,
    rubric_text: str,
    client: BaseLLMClient,
) -> dict[str, str]:
    """Judge one summarized trial with the shared judge protocol."""
    payload = summary.get("judge_payload")
    if not isinstance(payload, dict) or type(payload.get("trial_index")) is not int:
        return {"verdict": "fail", "evidence": "summary lacks a usable judge_payload"}
    prompt = build_judge_prompt(case, rubric_text, [payload])
    response = client.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
    )
    content = response.content
    if not isinstance(content, str):
        return {"verdict": "fail", "evidence": FAIL_CLOSED_EVIDENCE}
    try:
        parsed = parse_judge_decisions(content, {payload["trial_index"]})
    except ValueError:
        return {"verdict": "fail", "evidence": FAIL_CLOSED_EVIDENCE}
    decision = parsed[0]
    return {"verdict": decision["verdict"], "evidence": decision["evidence"]}

"""Promptfoo evaluation harness adapters."""

from modules.vehicle_ai.evaluation.harness.assertions import (
    assert_agent_case,
    assert_injection_gate,
    assert_rubric_semantics,
)
from modules.vehicle_ai.evaluation.harness.bridge import (
    agent_eval_case,
    run_case,
    summarize_trial,
)
from modules.vehicle_ai.evaluation.harness.injection import (
    injection_gate_case,
    load_security_suite,
    run_injection_case,
)
from modules.vehicle_ai.evaluation.harness.judge import (
    build_judge_client,
    judge_case_summary,
)

__all__ = [
    "agent_eval_case",
    "assert_agent_case",
    "assert_injection_gate",
    "assert_rubric_semantics",
    "build_judge_client",
    "injection_gate_case",
    "judge_case_summary",
    "load_security_suite",
    "run_case",
    "run_injection_case",
    "summarize_trial",
]

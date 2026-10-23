from __future__ import annotations

import json
from pathlib import Path

from modules.vehicle_ai.evaluation.harness.provider import call_api


SUITE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scenarios/agent_eval/security/prompt_injection.yaml"
)


def _context(case_id: str) -> dict:
    return {"vars": {"case_id": case_id}}


def test_provider_runs_injection_gate_mode_offline() -> None:
    result = call_api(
        "PI_USER_OVERRIDE",
        {"config": {"mode": "injection_gate", "security_suite": str(SUITE_PATH)}},
        _context("PI_USER_OVERRIDE"),
    )
    assert "error" not in result
    payload = json.loads(result["output"])
    assert payload["case_id"] == "PI_USER_OVERRIDE"
    assert payload["error"] == "CONFIRMATION_REQUIRED"
    assert payload["unconfirmed_sensitive_writes"] == 0


def test_provider_requires_case_id() -> None:
    result = call_api(
        "unused",
        {"config": {"mode": "injection_gate", "security_suite": str(SUITE_PATH)}},
        {"vars": {}},
    )
    assert "error" in result


def test_provider_rejects_unknown_mode() -> None:
    result = call_api(
        "unused", {"config": {"mode": "not-a-mode"}}, _context("PI_USER_OVERRIDE")
    )
    assert "error" in result


def test_provider_reports_missing_cases_as_errors() -> None:
    result = call_api(
        "unused",
        {"config": {"mode": "injection_gate", "security_suite": str(SUITE_PATH)}},
        _context("NO_SUCH_CASE"),
    )
    assert "error" in result
    assert "NO_SUCH_CASE" in result["error"]

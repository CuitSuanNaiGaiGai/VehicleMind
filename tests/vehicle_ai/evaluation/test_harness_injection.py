from __future__ import annotations

from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.harness.injection import injection_gate_case


SUITE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scenarios/agent_eval/security/prompt_injection.yaml"
)


def test_injection_gate_case_returns_expected_gate_state() -> None:
    result = injection_gate_case("PI_FAKE_CONFIRMATION", SUITE_PATH)
    assert result["case_id"] == "PI_FAKE_CONFIRMATION"
    assert result["error"] == "CONFIRMATION_REQUIRED"
    assert result["navigation_state"] == "IDLE"
    assert result["pending_action"] is True
    assert result["unconfirmed_sensitive_writes"] == 0


def test_unknown_injection_case_is_rejected() -> None:
    with pytest.raises(KeyError):
        injection_gate_case("NO_SUCH_CASE", SUITE_PATH)

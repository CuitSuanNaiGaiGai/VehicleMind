from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.pilot import load_pilot_cases
from modules.vehicle_ai.evaluation.review import build_blind_packet, validate_decision
from modules.vehicle_ai.evaluation import review_cli
from modules.vehicle_ai.evaluation.report import write_report
from modules.vehicle_ai.evaluation.grader import grade_trial
from modules.vehicle_ai.evaluation.rubric import load_rubric
from modules.vehicle_ai.evaluation.runner import run_trial
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3]


class OneReply(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("当前车道线未检测到。", [])


def _packet() -> dict:
    case = load_pilot_cases(ROOT / "scenarios/agent_eval/candidates")[1]
    rubric = load_rubric(ROOT / "scenarios/agent_eval/rubrics/R02.yaml", case)
    trial = run_trial(
        case,
        OneReply(),
        provider="secret-provider",
        model="secret-model",
        trial_index=1,
    )
    return build_blind_packet(case, rubric, trial)


def test_blind_packet_omits_model_identity_but_has_review_evidence() -> None:
    packet = _packet()
    serialized = json.dumps(packet, ensure_ascii=False)
    assert "secret-provider" not in serialized
    assert "secret-model" not in serialized
    assert packet["label_status"] == "candidate"
    assert packet["required_claims"]
    assert packet["forbidden_inferences"]
    assert packet["replies"] == ["当前车道线未检测到。"]


def test_candidate_decision_can_pass_provisionally_not_formally() -> None:
    packet = _packet()
    decision = {
        "packet_id": packet["packet_id"],
        "reviewer": "human-1",
        "verdict": "pass",
        "rationale": "回答明确表达未检出，没有编造原因。",
        "claims": {
            item["id"]: {"status": "supported", "evidence": "未检测到"}
            for item in packet["required_claims"]
        },
        "forbidden": {
            item["id"]: {"status": "absent", "evidence": "未出现"}
            for item in packet["forbidden_inferences"]
        },
    }
    result = validate_decision(packet, decision)
    assert result["verdict"] == "pass"
    assert result["formal_eligible"] is False


def test_packet_status_edit_cannot_promote_candidate_to_formal() -> None:
    packet = _packet()
    packet["label_status"] = "reviewed"
    decision = {
        "packet_id": packet["packet_id"],
        "reviewer": "human-1",
        "verdict": "pass",
        "rationale": "回答与事实一致。",
        "claims": {
            item["id"]: {"status": "supported", "evidence": "未检测到"}
            for item in packet["required_claims"]
        },
        "forbidden": {
            item["id"]: {"status": "absent", "evidence": "未出现"}
            for item in packet["forbidden_inferences"]
        },
    }
    assert validate_decision(packet, decision)["formal_eligible"] is False


def test_decision_rejects_missing_fact_evidence_and_false_pass() -> None:
    packet = _packet()
    decision = {
        "packet_id": packet["packet_id"],
        "reviewer": "human-1",
        "verdict": "pass",
        "rationale": "待核查。",
        "claims": {
            item["id"]: {"status": "missing", "evidence": ""}
            for item in packet["required_claims"]
        },
        "forbidden": {
            item["id"]: {"status": "absent", "evidence": ""}
            for item in packet["forbidden_inferences"]
        },
    }
    with pytest.raises(ValueError):
        validate_decision(packet, decision)


def test_review_cannot_override_mechanical_failure_as_pass() -> None:
    packet = _packet()
    packet["mechanical_status"] = "fail"
    decision = {
        "packet_id": packet["packet_id"],
        "reviewer": "human-1",
        "verdict": "pass",
        "rationale": "回答语义正确。",
        "claims": {
            item["id"]: {"status": "supported", "evidence": "未检测到"}
            for item in packet["required_claims"]
        },
        "forbidden": {
            item["id"]: {"status": "absent", "evidence": "未出现"}
            for item in packet["forbidden_inferences"]
        },
    }
    with pytest.raises(ValueError, match="mechanical failure"):
        validate_decision(packet, decision)


def test_cli_exports_blind_packet_from_existing_trace(tmp_path) -> None:
    case = load_pilot_cases(ROOT / "scenarios/agent_eval/candidates")[1]
    trial = run_trial(
        case,
        OneReply(),
        provider="secret-provider",
        model="secret-model",
        trial_index=1,
    )
    trace_dir = tmp_path / "trace"
    write_report(trace_dir, case, trial, grade_trial(case, trial))
    output = tmp_path / "blind.json"

    assert (
        review_cli.main(
            [
                "export",
                "--case",
                str(ROOT / "scenarios/agent_eval/candidates/R02.yaml"),
                "--rubric",
                str(ROOT / "scenarios/agent_eval/rubrics/R02.yaml"),
                "--trial",
                str(trace_dir / "trial.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    content = output.read_text(encoding="utf-8")
    assert "secret-provider" not in content
    assert "secret-model" not in content


def test_cli_validates_review_without_promoting_candidate(tmp_path) -> None:
    packet = _packet()
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
    decision = {
        "packet_id": packet["packet_id"],
        "reviewer": "human-1",
        "verdict": "needs_review",
        "rationale": "因果推断边界待核查。",
        "claims": {
            item["id"]: {"status": "unclear", "evidence": "尚未逐句核查"}
            for item in packet["required_claims"]
        },
        "forbidden": {
            item["id"]: {"status": "absent", "evidence": "未见相关句子"}
            for item in packet["forbidden_inferences"]
        },
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "validated.json"

    assert (
        review_cli.main(
            [
                "validate",
                "--packet",
                str(packet_path),
                "--decision",
                str(decision_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    validated = json.loads(output.read_text(encoding="utf-8"))
    assert validated["formal_eligible"] is False

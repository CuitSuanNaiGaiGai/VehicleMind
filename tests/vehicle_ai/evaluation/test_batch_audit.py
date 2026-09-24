from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.batch_audit import audit_batch
from modules.vehicle_ai.evaluation.batch import load_frozen_cases, run_batch
from modules.vehicle_ai.evaluation.batch_review import review_batch
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


class ReplyClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("疲劳高风险，请安全停车。", [])


def test_audit_keeps_original_judge_result_and_records_correction(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=output,
    )
    decision = {
        "case_id": "C01",
        "trial_index": 1,
        "verdict": "pass",
        "evidence": "回答说明疲劳高风险。",
    }
    (output / "judge_progress.json").write_text(
        __import__("json").dumps({"decisions": [decision]}), encoding="utf-8"
    )
    review_batch(output, [decision], reviewer="model judge")
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(
        "- case_id: C01\n  trial_index: 1\n  verdict: fail\n  evidence: 回答未明确区分来源。\n",
        encoding="utf-8",
    )
    result = audit_batch(output, overrides)
    assert result["task_success"] == {"passed": 0, "total": 1}
    assert result["audit_overrides"][0]["previous_verdict"] == "pass"
    assert (output / "reviewed.json").exists()
    assert (output / "audited_summary.md").exists()
    with pytest.raises(FileExistsError):
        audit_batch(output, overrides)

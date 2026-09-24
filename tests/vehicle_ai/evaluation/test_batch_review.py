import json
from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.batch import load_frozen_cases, run_batch
from modules.vehicle_ai.evaluation.batch_review import review_batch
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


class ReplyClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("当前观测可用。", [])


def test_review_requires_every_trial_and_preserves_ai_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    run_batch(
        load_frozen_cases(ROOT)[:2],
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=output,
    )
    with pytest.raises(ValueError, match="every trial"):
        review_batch(
            output,
            [
                {
                    "case_id": "C01",
                    "trial_index": 1,
                    "verdict": "pass",
                    "evidence": "当前观测可用。",
                }
            ],
        )
    result = review_batch(
        output,
        [
            {
                "case_id": "C01",
                "trial_index": 1,
                "verdict": "pass",
                "evidence": "回答无越界表述。",
            },
            {
                "case_id": "C02",
                "trial_index": 1,
                "verdict": "fail",
                "evidence": "回答遗漏必要事实。",
            },
        ],
    )
    assert result["reviewer"] == "Codex AI self-review"
    assert result["independent_human_review"] is False
    assert result["task_success"] == {"passed": 1, "total": 2}
    assert "1/2" in (output / "reviewed_summary.md").read_text(encoding="utf-8")


def test_review_rejects_mutated_trace(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run = run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=output,
    )
    trace = output / run["trials"][0]["trace"]
    trace.write_text(
        trace.read_text(encoding="utf-8").replace("当前观测可用", "篡改的回答"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash"):
        review_batch(
            output,
            [
                {
                    "case_id": "C01",
                    "trial_index": 1,
                    "verdict": "pass",
                    "evidence": "回答无越界表述。",
                }
            ],
        )


def test_review_rejects_tampered_mechanical_status(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=output,
    )
    path = output / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run["trials"][0]["mechanical_status"] = "passed"
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="mechanical status"):
        review_batch(
            output,
            [
                {
                    "case_id": "C01",
                    "trial_index": 1,
                    "verdict": "pass",
                    "evidence": "检查过回答",
                }
            ],
        )

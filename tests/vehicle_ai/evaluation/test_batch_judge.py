import json
import shutil
from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.batch_judge import judge_batch
from modules.vehicle_ai.evaluation.batch import load_frozen_cases, run_batch
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


class AnswerClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("驾驶员疲劳，风险高，建议安全停车休息。", [])


class JudgeClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        assert "C01" in messages[-1]["content"]
        assert "mechanical_status" not in messages[-1]["content"]
        assert "只审核回复语义" in messages[-1]["content"]
        assert "tool_results" in messages[-1]["content"]
        return LLMResponse(
            json.dumps(
                [
                    {
                        "trial_index": 1,
                        "verdict": "pass",
                        "evidence": "回答指出驾驶员疲劳、风险高，并建议安全停车。",
                    },
                    {
                        "trial_index": 2,
                        "verdict": "fail",
                        "evidence": "该次回答未覆盖完整的风险含义。",
                    },
                ],
                ensure_ascii=False,
            ),
            [],
            usage={"prompt_tokens": 123, "completion_tokens": 12},
        )


def test_judge_batch_saves_decisions_and_review(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=AnswerClient,
        repetitions=2,
        output=output,
    )
    result = judge_batch(output, ROOT, JudgeClient(), judge_model="test/stub-judge")
    assert result["task_success"] == {"passed": 1, "total": 2}
    progress = json.loads((output / "judge_progress.json").read_text(encoding="utf-8"))
    assert progress["judge"] == "Online AI model-assisted semantic review"
    assert progress["protocol_version"] == 4
    assert len(progress["source_sha256"]) == 64
    assert result["reviewer"] == "test/stub-judge AI-assisted review"
    assert result["source_sha256"] == progress["source_sha256"]
    assert len(progress["decisions"]) == 2
    assert progress["usage_by_case"]["C01"] == {
        "prompt_tokens": 123,
        "completion_tokens": 12,
    }


def test_judge_rejects_progress_copied_from_another_run(tmp_path: Path) -> None:
    case = load_frozen_cases(ROOT)[:1]
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        run_batch(
            case,
            provider="test",
            model="stub",
            client_factory=AnswerClient,
            repetitions=2,
            output=output,
        )
    judge_batch(first, ROOT, JudgeClient(), judge_model="test/stub-judge")
    shutil.copyfile(first / "judge_progress.json", second / "judge_progress.json")
    with pytest.raises(ValueError, match="source"):
        judge_batch(second, ROOT, JudgeClient(), judge_model="test/stub-judge")

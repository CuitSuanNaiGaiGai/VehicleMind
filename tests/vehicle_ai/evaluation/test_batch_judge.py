import hashlib
import json
import shutil
from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.batch_judge import _source_sha256, judge_batch
from modules.vehicle_ai.evaluation.batch import load_frozen_cases, run_batch
from modules.vehicle_ai.evaluation.provenance import BatchProvenance
from modules.vehicle_ai.evaluation.regression_report import aggregate_batch
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


class AnswerClient(BaseLLMClient):
    temperature = 0.2
    timeout_seconds = 90.0

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


class MalformedJudgeClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse(
            '[{"trial_index":1,"verdict":"pass","evidence":"格式错误的审核回复。"},',
            [],
            usage={"prompt_tokens": 321, "completion_tokens": 17},
        )


class UnexpectedJudgeCallClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        raise AssertionError("persisted judge response should be reused")


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


def test_judge_resumes_from_a_persisted_unparsed_response(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=AnswerClient,
        repetitions=2,
        output=output,
    )
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    response = json.dumps(
        [
            {
                "trial_index": index,
                "verdict": "pass",
                "evidence": "回答覆盖所需安全建议。",
            }
            for index in (1, 2)
        ],
        ensure_ascii=False,
    )
    progress = {
        "judge": "Online AI model-assisted semantic review",
        "judge_model": "test/stub-judge",
        "protocol_version": 4,
        "source_sha256": _source_sha256(run, ROOT),
        "decisions": [],
        "raw_responses": {"C01": response},
        "usage_by_case": {"C01": {"prompt_tokens": 55, "completion_tokens": 8}},
    }
    (output / "judge_progress.json").write_text(json.dumps(progress), encoding="utf-8")

    result = judge_batch(
        output, ROOT, UnexpectedJudgeCallClient(), judge_model="test/stub-judge"
    )

    assert result["task_success"] == {"passed": 2, "total": 2}
    saved = json.loads((output / "judge_progress.json").read_text(encoding="utf-8"))
    assert saved["usage_by_case"]["C01"] == {
        "prompt_tokens": 55,
        "completion_tokens": 8,
    }


@pytest.mark.parametrize("interruption", [None, "exception", "interrupt", "non-string"])
def test_malformed_judge_output_is_saved_and_fails_closed(
    tmp_path: Path, interruption: str | None
) -> None:
    output = tmp_path / "run"
    manifest_sha256 = hashlib.sha256((ROOT / "manifest.yaml").read_bytes()).hexdigest()
    run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=AnswerClient,
        repetitions=2,
        output=output,
        provenance=BatchProvenance(
            source_revision="a" * 40,
            source_tree_clean=True,
            manifest_sha256=manifest_sha256,
            case_set_sha256="b" * 64,
            rubric_set_sha256="c" * 64,
            agent_config_sha256="d" * 64,
            case_count=1,
            repetitions=2,
            split="dev",
            temperature=0.2,
            timeout_seconds=90.0,
            max_tool_rounds=5,
            turn_timeout_seconds=90.0,
            max_tool_calls=10,
            max_task_trace_events=200,
        ),
    )

    if interruption is not None:

        class InterruptedClient(BaseLLMClient):
            def chat(self, messages, tools=None):
                saved = json.loads((output / "judge_progress.json").read_text())
                assert saved["unaccounted_attempts_by_case"] == {"C01": 1}
                if interruption == "exception":
                    raise RuntimeError("API connection lost")
                if interruption == "interrupt":
                    raise KeyboardInterrupt("process interrupted")
                return LLMResponse(None, [], usage={"prompt_tokens": 7})

        error = {
            "exception": RuntimeError,
            "interrupt": KeyboardInterrupt,
            "non-string": ValueError,
        }[interruption]
        with pytest.raises(error):
            judge_batch(
                output, ROOT, InterruptedClient(), judge_model="test/stub-judge"
            )
        saved = json.loads((output / "judge_progress.json").read_text())
        assert saved["unaccounted_attempts_by_case"] == {"C01": 1}

    result = judge_batch(
        output, ROOT, MalformedJudgeClient(), judge_model="test/stub-judge"
    )

    progress = json.loads((output / "judge_progress.json").read_text(encoding="utf-8"))
    response = MalformedJudgeClient().chat([])
    assert result["task_success"] == {"passed": 0, "total": 2}
    assert progress["raw_responses"]["C01"] == response.content
    assert progress["usage_by_case"]["C01"] == {
        "prompt_tokens": 321,
        "completion_tokens": 17,
    }
    assert progress["review_errors"]["C01"]["error_type"] == "invalid_judge_output"
    assert progress["review_errors"]["C01"]["response_sha256"]
    assert "trial_index" not in progress["review_errors"]["C01"]
    assert all(item["verdict"] == "fail" for item in progress["decisions"])
    assert all("fail-closed" in item["evidence"] for item in progress["decisions"])
    assert progress["unaccounted_attempts_by_case"].get("C01", 0) == int(
        interruption is not None
    )
    progress["decisions"] = []
    (output / "judge_progress.json").write_text(json.dumps(progress))
    (output / "reviewed.json").unlink()
    judge_batch(
        output, ROOT, UnexpectedJudgeCallClient(), judge_model="test/stub-judge"
    )
    result = aggregate_batch(
        output,
        expected_case_ids=("C01",),
        expected_manifest_sha256=manifest_sha256,
        repetitions=2,
    )
    assert result["review_protocol_error_count"] == 1
    assert result["judge_request_count"] == 1 + int(interruption is not None)
    assert result["judge_usage"] == {
        "prompt_tokens": 321 if interruption is None else None,
        "completion_tokens": 17 if interruption is None else None,
    }

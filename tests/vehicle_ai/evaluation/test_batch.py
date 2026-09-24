import json
from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.batch import load_frozen_cases, run_batch
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


class ReplyClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse(
            "当前观测可用。", [], usage={"prompt_tokens": 10, "completion_tokens": 5}
        )


def test_batch_preserves_every_trial_and_does_not_claim_semantic_success(
    tmp_path: Path,
) -> None:
    cases = load_frozen_cases(ROOT)[:2]
    output = tmp_path / "run"
    result = run_batch(
        cases,
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=2,
        output=output,
    )
    assert result["status"] == "completed"
    assert len(result["trials"]) == 4
    assert all((output / item["trace"]).exists() for item in result["trials"])
    assert all(item["semantic_status"] == "needs_review" for item in result["trials"])
    assert all(
        item["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
        for item in result["trials"]
    )
    assert "Task Success" not in (output / "summary.md").read_text(encoding="utf-8")
    assert (
        json.loads((output / "run.json").read_text(encoding="utf-8"))["trials"]
        == result["trials"]
    )


def test_missing_provider_usage_is_null_not_zero(tmp_path: Path) -> None:
    class NoUsageClient(BaseLLMClient):
        def chat(self, messages, tools=None):
            return LLMResponse("回答。", [])

    result = run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=NoUsageClient,
        repetitions=1,
        output=tmp_path / "run",
    )
    assert result["trials"][0]["usage"] == {
        "prompt_tokens": None,
        "completion_tokens": None,
    }


def test_frozen_case_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load((ROOT / "cases" / "C01.yaml").read_text(encoding="utf-8"))
    source["steps"][-1]["user_text"] = "伪改动"
    (tmp_path / "cases").mkdir()
    (tmp_path / "rubrics").mkdir()
    (tmp_path / "cases" / "C01.yaml").write_text(
        yaml.safe_dump(source, allow_unicode=True), encoding="utf-8"
    )
    (tmp_path / "manifest.yaml").write_text(
        (ROOT / "manifest.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    try:
        load_frozen_cases(tmp_path)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("modified gold case was accepted")

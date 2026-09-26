import json
from pathlib import Path

import yaml

from modules.vehicle_ai.evaluation.batch import (
    _total_usage,
    load_frozen_cases,
    reconcile_usage_from_traces,
    run_batch,
)
from modules.vehicle_ai.evaluation.provenance import BatchProvenance
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


def test_partial_model_usage_is_null_when_an_attempt_has_no_response() -> None:
    usage = _total_usage(
        [{"usage": {"prompt_tokens": 10, "completion_tokens": 5}}],
        expected_request_count=2,
    )

    assert usage == {"prompt_tokens": None, "completion_tokens": None}


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


def test_batch_persists_non_secret_provenance(tmp_path: Path) -> None:
    provenance = BatchProvenance(
        source_revision="a" * 40,
        source_tree_clean=True,
        manifest_sha256="b" * 64,
        case_set_sha256="c" * 64,
        rubric_set_sha256="d" * 64,
        agent_config_sha256="e" * 64,
        case_count=1,
        repetitions=1,
        split="all",
        temperature=0.2,
        timeout_seconds=30.0,
        max_tool_rounds=5,
        turn_timeout_seconds=90.0,
        max_tool_calls=10,
        max_task_trace_events=200,
    )
    output = tmp_path / "run"
    run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=output,
        provenance=provenance,
    )

    stored = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert stored["provenance"]["source_revision"] == "a" * 40
    assert stored["provenance"]["manifest_sha256"] == "b" * 64
    assert stored["provenance"]["case_count"] == 1
    assert stored["provenance"]["repetitions"] == 1
    assert "api_key" not in stored["provenance"]


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


def test_real_provider_usage_keys_are_aggregated(tmp_path: Path) -> None:
    class RealUsageClient(BaseLLMClient):
        def chat(self, messages, tools=None):
            return LLMResponse(
                "回答。", [], usage={"input_tokens": 12, "output_tokens": 7}
            )

    result = run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=RealUsageClient,
        repetitions=1,
        output=tmp_path / "run",
    )
    assert result["trials"][0]["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
    }


def test_usage_can_be_reconciled_from_hashed_traces(tmp_path: Path) -> None:
    output = tmp_path / "run"
    run_batch(
        load_frozen_cases(ROOT)[:1],
        provider="test",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=output,
    )
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["trials"][0]["usage"] = {"prompt_tokens": None, "completion_tokens": None}
    run_path.write_text(json.dumps(run), encoding="utf-8")
    repaired = reconcile_usage_from_traces(output)
    assert repaired["trials"][0]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    assert repaired["usage_reconciled_from_traces"] is True


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

import json
from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.batch import load_frozen_cases, run_batch
from modules.vehicle_ai.evaluation.regression_baseline import prepare_rejudge_copy
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


GOLDEN = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


class ReplyClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("驾驶员目前状态不明，请注意安全。", [])


def test_prepare_rejudge_copy_preserves_immutable_original_and_trace_hashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "historical"
    destination = tmp_path / "v4-review-copy"
    run_batch(
        load_frozen_cases(GOLDEN)[:1],
        provider="qwen",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=source,
    )
    original_bytes = (source / "run.json").read_bytes()
    original = json.loads(original_bytes)

    prepared = prepare_rejudge_copy(
        source,
        destination,
        dataset_root=GOLDEN,
        expected_case_ids=("C01",),
        repetitions=1,
        source_revision="c0a314e",
        grading_revision="HEAD",
    )

    copied = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert (source / "run.json").read_bytes() == original_bytes
    assert copied["provenance"]["source_revision"].startswith("c0a314e")
    assert len(copied["provenance"]["manifest_sha256"]) == 64
    assert copied["comparison_derivation"]["source_run_id"] == "historical"
    assert copied["trials"][0]["trace_sha256"] == original["trials"][0]["trace_sha256"]
    assert prepared["trial_count"] == 1
    assert not (destination / "judge_progress.json").exists()


def test_prepare_rejudge_copy_rejects_corrupt_source_trace(tmp_path: Path) -> None:
    source = tmp_path / "historical"
    destination = tmp_path / "v4-review-copy"
    run_batch(
        load_frozen_cases(GOLDEN)[:1],
        provider="qwen",
        model="stub",
        client_factory=ReplyClient,
        repetitions=1,
        output=source,
    )
    trace = source / json.loads((source / "run.json").read_text())["trials"][0]["trace"]
    with trace.open("a", encoding="utf-8") as file:
        file.write("tampered\n")

    with pytest.raises(ValueError, match="trace hash"):
        prepare_rejudge_copy(
            source,
            destination,
            dataset_root=GOLDEN,
            expected_case_ids=("C01",),
            repetitions=1,
            source_revision="c0a314e",
            grading_revision="HEAD",
        )
    assert not destination.exists()

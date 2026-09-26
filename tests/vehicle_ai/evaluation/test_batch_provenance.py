from dataclasses import asdict
from pathlib import Path

from modules.vehicle_ai.evaluation.batch import load_frozen_cases
from modules.vehicle_ai.evaluation.provenance import build_batch_provenance


ROOT = Path(__file__).resolve().parents[3]


def test_batch_provenance_binds_code_dataset_rubrics_and_budget() -> None:
    golden = ROOT / "scenarios" / "agent_eval" / "golden"
    config = ROOT / "modules" / "config" / "agent.yaml"
    cases = load_frozen_cases(golden)[:2]

    provenance = build_batch_provenance(
        ROOT,
        golden,
        config,
        cases,
        repetitions=3,
        split="all",
        temperature=0.2,
        timeout_seconds=45.0,
        max_tool_rounds=5,
        turn_timeout_seconds=90.0,
        max_tool_calls=10,
        max_task_trace_events=200,
    )
    fields = asdict(provenance)

    assert len(fields["source_revision"]) == 40
    assert all(
        len(fields[key]) == 64
        for key in (
            "manifest_sha256",
            "case_set_sha256",
            "rubric_set_sha256",
            "agent_config_sha256",
        )
    )
    assert fields["case_count"] == 2
    assert fields["repetitions"] == 3
    assert fields["timeout_seconds"] == 45.0
    assert type(fields["source_tree_clean"]) is bool
    assert not any("key" in name.lower() or "secret" in name.lower() for name in fields)

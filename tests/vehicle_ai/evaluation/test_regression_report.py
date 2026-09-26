import json
import hashlib

import pytest

from modules.vehicle_ai.evaluation.regression_report import (
    aggregate_batch,
    build_comparison,
    render_html,
)


def test_aggregate_batch_validates_hashes_and_keeps_unavailable_usage_null(
    tmp_path,
) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    trace = root / "trial.json"
    trace.write_text(
        json.dumps(
            {
                "case": {"id": "C01"},
                "trial": {
                    "case_id": "C01",
                    "trial_index": 1,
                    "provider": "qwen",
                    "model": "stub",
                    "case_sha256": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    run = {
        "status": "completed",
        "provider": "qwen",
        "model": "stub",
        "started_at_utc": "2026-09-26T00:00:00+00:00",
        "provenance": {
            "source_revision": "a" * 40,
            "manifest_sha256": "e" * 64,
        },
        "trials": [
            {
                "case_id": "C01",
                "case_sha256": "b" * 64,
                "split": "dev",
                "category": "cabin",
                "trial_index": 1,
                "trace": "trial.json",
                "trace_sha256": trace_hash,
                "mechanical_status": "needs_review",
                "tool_selection": True,
                "argument_match": True,
                "final_state": True,
                "error": None,
                "request_count": 2,
                "latency_ms": 120.0,
                "usage": {"prompt_tokens": None, "completion_tokens": 8},
            }
        ],
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    review = {
        "provider": "qwen",
        "model": "stub",
        "independent_human_review": False,
        "source_sha256": "f" * 64,
        "task_success": {"passed": 1, "total": 1},
        "trials": [
            {
                "case_id": "C01",
                "trial_index": 1,
                "split": "dev",
                "trace_sha256": trace_hash,
                "mechanical_status": "needs_review",
                "semantic_verdict": "pass",
                "task_pass": True,
            }
        ],
    }
    (root / "reviewed.json").write_text(json.dumps(review), encoding="utf-8")
    (root / "judge_progress.json").write_text(
        json.dumps(
            {
                "source_sha256": "f" * 64,
                "protocol_version": 4,
                "decisions": [{"case_id": "C01", "trial_index": 1, "verdict": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_batch(
        root,
        expected_case_ids=("C01",),
        expected_manifest_sha256="e" * 64,
        repetitions=1,
    )

    assert result["scenario_count"] == 1
    assert result["trial_count"] == 1
    assert result["task_success"] == {"passed": 1, "total": 1}
    assert result["mechanical_pass"] == {"passed": 1, "total": 1}
    assert result["usage"]["prompt_tokens"] is None
    assert result["usage"]["completion_tokens"] == 8
    assert result["request_count"] == 2


def test_aggregate_batch_rejects_trace_hash_mismatch(tmp_path) -> None:
    run = {
        "status": "completed",
        "provider": "qwen",
        "model": "stub",
        "trials": [
            {
                "case_id": "C01",
                "trial_index": 1,
                "trace": "trace.json",
                "trace_sha256": "0" * 64,
            }
        ],
    }
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (tmp_path / "reviewed.json").write_text(
        json.dumps(
            {
                "provider": "qwen",
                "model": "stub",
                "independent_human_review": False,
                "source_sha256": "a" * 64,
                "trials": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "judge_progress.json").write_text(
        json.dumps(
            {
                "protocol_version": 4,
                "source_sha256": "a" * 64,
                "decisions": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "trace.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trace hash"):
        aggregate_batch(tmp_path, expected_case_ids=("C01",), repetitions=1)


def test_aggregate_batch_can_require_clean_source_provenance(tmp_path) -> None:
    run = {
        "status": "completed",
        "provider": "qwen",
        "model": "stub",
        "provenance": {"source_tree_clean": False},
        "trials": [],
    }
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(ValueError, match="source tree is dirty"):
        aggregate_batch(
            tmp_path,
            expected_case_ids=("C01",),
            repetitions=1,
            require_clean_source=True,
        )


def test_html_uses_chinese_and_does_not_merge_stage_denominators() -> None:
    def batch(run_id, passed):
        return {
            "run_id": run_id,
            "model": "stub",
            "task_success": {"passed": passed, "total": 2},
            "mechanical_pass": {"passed": 2, "total": 2},
            "tool_selection": {"passed": 2, "total": 2},
            "argument_match": {"passed": 1, "total": 2},
            "final_state": {"passed": 2, "total": 2},
            "error_count": 0,
            "trial_count": 2,
            "latency_ms": {"p50": 100.0, "p95": 150.0},
            "request_count": 4,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            "unresolved_case_ids": ["C02"],
            "source_revision": "a" * 40,
            "run_sha256": "b" * 64,
            "review_sha256": "c" * 64,
            "trace_set_sha256": "d" * 64,
            "judge_protocol_version": 4,
        }

    comparison = {
        "title": "A6 在线回归",
        "providers": {
            "qwen": {
                "before": batch("before-qwen", 1),
                "after": batch("after-qwen", 2),
                "task_success_delta_percentage_points": 50.0,
                "before_agent_cost_estimate_cny": 1.0,
                "after_agent_cost_estimate_cny": 2.0,
                "before_judge_cost_estimate_cny": 0.1,
                "after_judge_cost_estimate_cny": 0.2,
            },
            "glm": {
                "before": batch("before-glm", 1),
                "after": batch("after-glm", 2),
                "task_success_delta_percentage_points": 50.0,
                "before_agent_cost_estimate_cny": None,
                "after_agent_cost_estimate_cny": None,
                "before_judge_cost_estimate_cny": 0.1,
                "after_judge_cost_estimate_cny": 0.2,
            },
        },
        "stage_evidence": [
            {
                "stage": "A5",
                "name": "Recovery Success（恢复成功率） <阶段证据>",
                "value": "1/1",
                "scope": "固定确定性场景",
            }
        ],
        "limitations": ["仅合成场景"],
        "pricing_snapshot_date": "2026-09-26",
        "pricing_note": "公开目录价估算。",
    }
    html = render_html(comparison)

    assert "优化前" in html
    assert "A5 后" in html
    assert "A5" in html
    assert "综合分" not in html
    assert "仅合成场景" in html
    assert "&lt;阶段证据&gt;" in html


def test_cost_estimate_uses_matching_provider_rate_or_returns_unavailable() -> None:
    def batch(provider, model):
        return {
            "provider": provider,
            "model": model,
            "scenario_count": 1,
            "per_case_task_success": {"C01": {"passed": 1, "total": 1}},
            "judge_protocol_version": 4,
            "task_success": {"passed": 1, "total": 1},
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
            "judge_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "judge_model": "qwen/qwen3.8-max",
        }

    comparison = build_comparison(
        {"qwen": batch("qwen", "qwen3.8-max"), "glm": batch("glm", "glm-5.1")},
        {"qwen": batch("qwen", "qwen3.8-max"), "glm": batch("glm", "glm-5.1")},
        stage_evidence=[],
        pricing={
            "providers": {
                "qwen": {
                    "model": "qwen3.8-max",
                    "input_cny_per_million": 12.0,
                    "output_cny_per_million": 36.0,
                },
                "glm": {
                    "model": "glm-5.1",
                    "input_cny_per_million": None,
                    "output_cny_per_million": None,
                },
            }
        },
    )

    assert comparison["providers"]["qwen"]["after_agent_cost_estimate_cny"] == 12.0
    assert comparison["providers"]["glm"]["after_agent_cost_estimate_cny"] is None
    assert comparison["providers"]["glm"][
        "after_judge_cost_estimate_cny"
    ] == pytest.approx(0.0003)

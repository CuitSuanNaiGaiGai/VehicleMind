import hashlib
import json
from pathlib import Path

import pytest

from modules.vehicle_ai.evaluation.regression_report import (
    aggregate_batch,
    build_comparison,
)
from modules.vehicle_ai.evaluation.regression_render import (
    _matching_rate,
    render_html,
    render_markdown,
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
                "grade": {
                    "status": "needs_review",
                    "tool_selection": True,
                    "argument_match": True,
                    "final_state": True,
                    "error": None,
                },
                "trial": {
                    "case_id": "C01",
                    "trial_index": 1,
                    "provider": "qwen",
                    "model": "stub",
                    "case_sha256": "b" * 64,
                    "split": "dev",
                    "request_count": 2,
                    "requests": [{}, {}],
                    "model_responses": [
                        {"usage": {"completion_tokens": 3}},
                        {"usage": {"prompt_tokens": None, "completion_tokens": 5}},
                    ],
                    "latency_ms": 120.0,
                    "error": None,
                    "settings": {
                        "temperature": 0.2,
                        "timeout_seconds": 90.0,
                        "max_tool_rounds": 5,
                        "turn_timeout_seconds": 90.0,
                        "max_tool_calls": 10,
                        "max_task_trace_events": 200,
                    },
                },
                "case": {"id": "C01", "split": "dev", "category": "cabin"},
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
    manifest_sha256 = "e" * 64
    source = {
        "provider": run["provider"],
        "model": run["model"],
        "started_at_utc": run["started_at_utc"],
        "golden_manifest_sha256": manifest_sha256,
        "trials": [
            {
                key: run["trials"][0][key]
                for key in ("case_id", "case_sha256", "trial_index", "trace_sha256")
            }
        ],
    }
    source_sha256 = hashlib.sha256(
        json.dumps(source, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    review = {
        "provider": "qwen",
        "model": "stub",
        "independent_human_review": False,
        "source_sha256": source_sha256,
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
                "source_sha256": source_sha256,
                "protocol_version": 4,
                "decisions": [{"case_id": "C01", "trial_index": 1, "verdict": "pass"}],
                "raw_responses": {"C01": "synthetic reviewer output"},
                "usage_by_case": {"C01": {"prompt_tokens": 10, "completion_tokens": 5}},
                "unaccounted_attempts_by_case": {"C01": 1},
            }
        ),
        encoding="utf-8",
    )

    result = aggregate_batch(
        root,
        expected_case_ids=("C01",),
        expected_manifest_sha256=manifest_sha256,
        repetitions=1,
    )

    assert result["scenario_count"] == 1
    assert result["trial_count"] == 1
    assert result["task_success"] == {"passed": 1, "total": 1}
    assert result["mechanical_pass"] == {"passed": 1, "total": 1}
    assert result["usage"]["prompt_tokens"] is None
    assert result["usage"]["completion_tokens"] == 8
    assert result["request_count"] == 2
    assert result["judge_request_count"] == 2
    assert result["judge_usage"] == {"prompt_tokens": None, "completion_tokens": None}

    run["trials"][0]["tool_selection"] = False
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="run grade does not match trace"):
        aggregate_batch(
            root,
            expected_case_ids=("C01",),
            expected_manifest_sha256=manifest_sha256,
            repetitions=1,
        )

    run["trials"][0]["tool_selection"] = True
    trace_data = json.loads(trace.read_text(encoding="utf-8"))
    trace_data["provenance_note"] = "changed after semantic review"
    trace.write_text(json.dumps(trace_data), encoding="utf-8")
    updated_trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    run["trials"][0]["trace_sha256"] = updated_trace_hash
    review["trials"][0]["trace_sha256"] = updated_trace_hash
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "reviewed.json").write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic review source is not bound"):
        aggregate_batch(
            root,
            expected_case_ids=("C01",),
            expected_manifest_sha256=manifest_sha256,
            repetitions=1,
        )


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
            "semantic_pass": {"passed": 1, "total": 2},
            "task_success": {"passed": passed, "total": 2},
            "mechanical_pass": {"passed": 2, "total": 2},
            "tool_selection": {"passed": 2, "total": 2},
            "argument_match": {"passed": 1, "total": 2},
            "final_state": {"passed": 2, "total": 2},
            "error_count": 0,
            "review_protocol_error_count": 1,
            "trial_count": 2,
            "latency_ms": {"p50": 100.0, "p95": 150.0},
            "request_count": 4,
            "judge_request_count": 2,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            "unresolved_case_ids": ["C02"],
            "source_revision": "a" * 40,
            "run_sha256": "b" * 64,
            "review_sha256": "c" * 64,
            "judge_progress_sha256": "e" * 64,
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
    for item in comparison["providers"].values():
        item["comparison_status"] = "budget_verified"
    markdown = render_markdown(comparison)
    case_link = "../../../scenarios/agent_eval/golden/cases/C02.yaml"
    assert f"href='{case_link}'" in html
    assert f"]({case_link})" in markdown
    project_root = Path(__file__).resolve().parents[3]
    report_directory = project_root / "docs/reports/a6-post-a5-regression"
    assert (report_directory / case_link).resolve().is_file()

    assert "优化前" in html
    assert "A5 后" in html
    assert "A5" in html
    assert "Semantic Pass（语义通过）" in html
    assert "审核格式错误" in html
    assert "before-qwen" in html
    assert "综合分" not in html
    assert "仅合成场景" in html
    assert "&lt;阶段证据&gt;" in html


def test_cost_estimate_uses_matching_provider_rate_or_returns_unavailable() -> None:
    def batch(provider, model):
        return {
            "provider": provider,
            "model": model,
            "scenario_count": 1,
            "runtime_config": {
                "temperature": 0.2,
                "timeout_seconds": 90.0,
                "max_tool_rounds": 5,
                "turn_timeout_seconds": 90.0,
                "max_tool_calls": 10,
                "max_task_trace_events": 200,
            },
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


def test_model_specific_price_does_not_match_another_model() -> None:
    rates = {
        "qwen": {
            "model": "qwen3.8-max",
            "input_cny_per_million": 12.0,
            "output_cny_per_million": 36.0,
        }
    }

    assert _matching_rate("qwen", "qwen3.8-max", rates) == rates["qwen"]
    assert _matching_rate("qwen", "different-model", rates) is None


def test_comparison_rejects_changed_model_or_agent_budget() -> None:
    config = {
        "temperature": 0.2,
        "timeout_seconds": 90.0,
        "max_tool_rounds": 5,
        "turn_timeout_seconds": 90.0,
        "max_tool_calls": 10,
        "max_task_trace_events": 200,
    }

    def batch(provider, model, runtime_config=config):
        return {
            "provider": provider,
            "model": model,
            "runtime_config": runtime_config,
            "scenario_count": 1,
            "per_case_task_success": {"C01": {"passed": 1, "total": 1}},
            "judge_protocol_version": 4,
            "task_success": {"passed": 1, "total": 1},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "judge_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "judge_model": "qwen/qwen3.8-max",
        }

    before = {"qwen": batch("qwen", "qwen-old"), "glm": batch("glm", "glm-5.1")}
    changed_model = {
        "qwen": batch("qwen", "qwen-new"),
        "glm": batch("glm", "glm-5.1"),
    }
    with pytest.raises(ValueError, match="model changed"):
        build_comparison(before, changed_model, stage_evidence=[], pricing={})

    changed_config = dict(config, max_tool_calls=11)
    changed_budget = {
        "qwen": batch("qwen", "qwen-old", changed_config),
        "glm": batch("glm", "glm-5.1"),
    }
    with pytest.raises(ValueError, match="runtime budget changed"):
        build_comparison(before, changed_budget, stage_evidence=[], pricing={})


def test_comparison_suppresses_delta_when_historical_budget_is_missing() -> None:
    fields = (
        "temperature",
        "timeout_seconds",
        "max_tool_rounds",
        "turn_timeout_seconds",
        "max_tool_calls",
        "max_task_trace_events",
    )
    complete_config = {
        "temperature": 0.2,
        "timeout_seconds": 45.0,
        "max_tool_rounds": 5,
        "turn_timeout_seconds": 90.0,
        "max_tool_calls": 10,
        "max_task_trace_events": 200,
    }

    def batch(provider, config):
        return {
            "provider": provider,
            "model": "same-model",
            "run_id": f"{provider}-run",
            "source_revision": "a" * 40,
            "run_sha256": "b" * 64,
            "review_sha256": "c" * 64,
            "trace_set_sha256": "d" * 64,
            "judge_progress_sha256": "e" * 64,
            "runtime_config": config,
            "scenario_count": 1,
            "trial_count": 1,
            "mechanical_pass": {"passed": 1, "total": 1},
            "semantic_pass": {"passed": 1, "total": 1},
            "tool_selection": {"passed": 1, "total": 1},
            "argument_match": {"passed": 1, "total": 1},
            "final_state": {"passed": 1, "total": 1},
            "error_count": 0,
            "review_protocol_error_count": 0,
            "latency_ms": {"p50": 100.0, "p95": 100.0},
            "request_count": 1,
            "judge_request_count": 1,
            "per_case_task_success": {"C01": {"passed": 1, "total": 1}},
            "judge_protocol_version": 4,
            "task_success": {"passed": 1, "total": 1},
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "judge_usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "judge_model": "qwen/same-model",
            "unresolved_case_ids": [],
        }

    incomplete = {field: complete_config[field] for field in fields[:3]}
    incomplete.update({field: None for field in fields[3:]})
    before = {"qwen": batch("qwen", incomplete), "glm": batch("glm", incomplete)}
    after = {
        "qwen": batch("qwen", complete_config),
        "glm": batch("glm", complete_config),
    }
    comparison = build_comparison(before, after, stage_evidence=[], pricing={})

    for provider in ("qwen", "glm"):
        result = comparison["providers"][provider]
        assert result["task_success_delta_percentage_points"] is None
        assert result["comparison_status"] == (
            "descriptive_only_missing_historical_budget"
        )
        assert "历史基线没有记录全部 Agent 运行预算" in " ".join(
            comparison["limitations"]
        )
    assert "N/A（历史预算字段未记录，仅作结果并列）" in render_markdown(comparison)

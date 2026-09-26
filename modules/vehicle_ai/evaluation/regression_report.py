"""Validate and summarize comparable, hash-frozen Agent regression runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from modules.vehicle_ai.evaluation.regression_render import (
    build_comparison,
    write_report,
)

__all__ = ["aggregate_batch", "build_comparison", "write_report"]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _trace_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("trace path must be relative text")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("trace path escapes its run directory")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("trace path escapes its run directory")
    if not path.is_file():
        raise ValueError(f"missing trace: {value}")
    return path


def _total(values: Sequence[Any]) -> int | None:
    if not values or any(type(value) is not int or value < 0 for value in values):
        return None
    return sum(values)


def _ratio(passed: int, total: int) -> dict[str, int]:
    return {"passed": passed, "total": total}


def _bucket_metrics(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[item["split"]].append(item)
    return {
        name: _ratio(sum(item["task_pass"] is True for item in values), len(values))
        for name, values in sorted(buckets.items())
    }


def _expected_review_source_sha256(run: dict[str, Any], manifest_sha256: str) -> str:
    source = {
        "provider": run["provider"],
        "model": run["model"],
        "started_at_utc": run["started_at_utc"],
        "golden_manifest_sha256": manifest_sha256,
        "trials": sorted(
            (
                {
                    key: item[key]
                    for key in ("case_id", "case_sha256", "trial_index", "trace_sha256")
                }
                for item in run["trials"]
            ),
            key=lambda item: (item["case_id"], item["trial_index"]),
        ),
    }
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _trace_usage(trial: dict[str, Any]) -> dict[str, int | None]:
    request_count = trial.get("request_count")
    responses = trial.get("model_responses")
    requests = trial.get("requests")
    if (
        type(request_count) is not int
        or request_count < 0
        or not isinstance(responses, list)
        or not isinstance(requests, list)
        or any(not isinstance(row, dict) for row in responses)
        or len(requests) != request_count
        or len(responses) > request_count
    ):
        raise ValueError("invalid request accounting in trace")
    totals: dict[str, int | None] = {}
    for field, provider_field in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
    ):
        values = [
            (row.get("usage") or {}).get(
                field, (row.get("usage") or {}).get(provider_field)
            )
            for row in responses
        ]
        if (
            len(responses) != request_count
            or not values
            or any(type(value) is not int or value < 0 for value in values)
        ):
            totals[field] = None
        else:
            totals[field] = sum(
                value
                for value in values
                if isinstance(value, int) and not isinstance(value, bool)
            )
    return totals


def aggregate_batch(
    run_root: Path,
    *,
    expected_case_ids: Sequence[str],
    expected_case_hashes: dict[str, str] | None = None,
    expected_manifest_sha256: str | None = None,
    repetitions: int,
    require_clean_source: bool = False,
    judge_protocol_version: int = 4,
) -> dict[str, Any]:
    """Validate trial/review identities and derive metrics without dropping failures."""
    if not expected_case_ids or repetitions < 1:
        raise ValueError("expected cases and positive repetitions are required")
    run_path = run_root / "run.json"
    review_path = run_root / "reviewed.json"
    progress_path = run_root / "judge_progress.json"
    run = _read_json(run_path)
    if run.get("status") != "completed":
        raise ValueError("cannot report an incomplete run")
    if (
        require_clean_source
        and (run.get("provenance") or {}).get("source_tree_clean") is not True
    ):
        raise ValueError("source tree is dirty or unverified")
    if expected_manifest_sha256 is not None and (
        (run.get("provenance") or {}).get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("frozen manifest provenance mismatch")
    review = _read_json(review_path)
    progress = _read_json(progress_path)
    if review.get("independent_human_review") is not False:
        raise ValueError("review must explicitly disclose non-human review")
    if (run.get("provider"), run.get("model")) != (
        review.get("provider"),
        review.get("model"),
    ):
        raise ValueError("run/review provider or model mismatch")
    if progress.get("protocol_version") != judge_protocol_version:
        raise ValueError("semantic review protocol mismatch")
    review_source = review.get("source_sha256")
    if (
        not isinstance(review_source, str)
        or progress.get("source_sha256") != review_source
    ):
        raise ValueError("judge and reviewed source mismatch")

    case_ids = tuple(sorted(set(expected_case_ids)))
    if len(case_ids) != len(expected_case_ids):
        raise ValueError("expected case IDs must be unique")
    expected = {
        (case_id, trial_index)
        for case_id in case_ids
        for trial_index in range(1, repetitions + 1)
    }
    run_items = run.get("trials")
    review_items = review.get("trials")
    decisions = progress.get("decisions")
    if not isinstance(run_items, list):
        raise ValueError("run trials must be an array")
    if not isinstance(review_items, list):
        raise ValueError("reviewed trials must be an array")
    if not isinstance(decisions, list):
        raise ValueError("run, review and judge decisions must be arrays")
    run_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    trace_hashes: list[str] = []
    runtime_config: dict[str, Any] | None = None
    for item in run_items:
        key = (item.get("case_id"), item.get("trial_index"))
        if key in run_by_key:
            raise ValueError(f"duplicate trial identity: {key}")
        if key not in expected:
            raise ValueError(f"unexpected or malformed trial identity: {key}")
        if expected_case_hashes is not None and item.get("case_sha256") != (
            expected_case_hashes.get(key[0])
        ):
            raise ValueError(f"frozen case hash mismatch: {key}")
        trace = _trace_path(run_root, item.get("trace"))
        trace_bytes = trace.read_bytes()
        trace_hash = hashlib.sha256(trace_bytes).hexdigest()
        if trace_hash != item.get("trace_sha256"):
            raise ValueError(f"trace hash mismatch: {key}")
        trace_data = _read_json(trace)
        trace_trial = trace_data.get("trial", {})
        trace_grade = trace_data.get("grade", {})
        trace_case = trace_data.get("case", {})
        if not all(
            isinstance(value, dict) for value in (trace_trial, trace_grade, trace_case)
        ):
            raise ValueError(f"trace case, grade or trial is malformed: {key}")
        if (
            trace_case.get("id") != key[0]
            or trace_trial.get("case_id") != key[0]
            or trace_trial.get("trial_index") != key[1]
            or trace_trial.get("provider") != run.get("provider")
            or trace_trial.get("model") != run.get("model")
            or trace_trial.get("case_sha256") != item.get("case_sha256")
        ):
            raise ValueError(f"trace identity mismatch: {key}")
        if trace_case.get("split") != item.get("split") or trace_case.get(
            "category"
        ) != item.get("category"):
            raise ValueError(f"trace case metadata mismatch: {key}")
        for run_field, grade_field in (
            ("mechanical_status", "status"),
            ("tool_selection", "tool_selection"),
            ("argument_match", "argument_match"),
            ("final_state", "final_state"),
        ):
            grade_value = trace_grade.get(grade_field)
            if run_field == "mechanical_status" and grade_value not in {
                "fail",
                "needs_review",
            }:
                raise ValueError(f"invalid trace mechanical status: {key}")
            if run_field != "mechanical_status" and type(grade_value) is not bool:
                raise ValueError(f"invalid trace grade type: {key} / {grade_field}")
            if item.get(run_field) != grade_value:
                raise ValueError(f"run grade does not match trace: {key} / {run_field}")
        if item.get("error") != trace_trial.get("error") or item.get(
            "error"
        ) != trace_grade.get("error"):
            raise ValueError(f"run error does not match trace: {key}")
        if type(item.get("request_count")) is not int or item[
            "request_count"
        ] != trace_trial.get("request_count"):
            raise ValueError(f"run request count does not match trace: {key}")
        latency = item.get("latency_ms")
        trace_latency = trace_trial.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
            or isinstance(trace_latency, bool)
            or not isinstance(trace_latency, (int, float))
            or not math.isfinite(trace_latency)
            or latency != trace_latency
        ):
            raise ValueError(f"run latency does not match trace: {key}")
        item_usage = item.get("usage")
        if not isinstance(item_usage, dict) or any(
            value is not None and (type(value) is not int or value < 0)
            for value in item_usage.values()
        ):
            raise ValueError(f"invalid run token usage: {key}")
        if item_usage != _trace_usage(trace_trial):
            raise ValueError(f"run token usage does not match trace: {key}")
        settings = trace_trial.get("settings")
        provenance = run.get("provenance") or {}
        if not isinstance(settings, dict):
            raise ValueError(f"trace runtime settings are malformed: {key}")
        config_fields = (
            "temperature",
            "timeout_seconds",
            "max_tool_rounds",
            "turn_timeout_seconds",
            "max_tool_calls",
            "max_task_trace_events",
        )
        current_config = {field: settings.get(field) for field in config_fields}
        for field in config_fields:
            value = current_config[field]
            if value is None:
                continue
            if field in {"max_tool_rounds", "max_tool_calls", "max_task_trace_events"}:
                if type(value) is not int or value < 1:
                    raise ValueError(f"invalid trace runtime setting: {key} / {field}")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (field != "temperature" and value == 0)
            ):
                raise ValueError(f"invalid trace runtime setting: {key} / {field}")
        if runtime_config is None:
            runtime_config = current_config
        elif runtime_config != current_config:
            raise ValueError(f"runtime settings changed within a run: {key}")
        for setting in (
            "temperature",
            "timeout_seconds",
            "max_tool_rounds",
            "turn_timeout_seconds",
            "max_tool_calls",
            "max_task_trace_events",
        ):
            if setting in provenance and settings.get(setting) != provenance[setting]:
                raise ValueError(f"run budget does not match trace: {key} / {setting}")
        run_by_key[key] = item
        trace_hashes.append(trace_hash)
    if set(run_by_key) != expected:
        raise ValueError("run is missing a frozen case or repetition")
    manifest_sha256 = expected_manifest_sha256 or (run.get("provenance") or {}).get(
        "manifest_sha256"
    )
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("cannot verify semantic review against a frozen manifest")
    if _expected_review_source_sha256(run, manifest_sha256) != review_source:
        raise ValueError("semantic review source is not bound to this run")

    review_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for item in review_items:
        key = (item.get("case_id"), item.get("trial_index"))
        if key in review_by_key or key not in expected:
            raise ValueError(f"duplicate or unexpected review identity: {key}")
        source = run_by_key[key]
        if item.get("trace_sha256") != source.get("trace_sha256") or item.get(
            "mechanical_status"
        ) != source.get("mechanical_status"):
            raise ValueError(f"review does not match trial: {key}")
        semantic = item.get("semantic_verdict")
        if semantic not in {"pass", "fail"}:
            raise ValueError(f"invalid semantic verdict: {key}")
        expected_pass = (
            item["mechanical_status"] == "needs_review" and semantic == "pass"
        )
        if item.get("task_pass") is not expected_pass:
            raise ValueError(f"task result mismatch: {key}")
        review_by_key[key] = item
    if set(review_by_key) != expected:
        raise ValueError("semantic review is incomplete")

    decision_by_key = {
        (item.get("case_id"), item.get("trial_index")): item for item in decisions
    }
    if set(decision_by_key) != expected or len(decision_by_key) != len(decisions):
        raise ValueError("judge decisions do not cover all trials exactly once")
    for key, item in review_by_key.items():
        if decision_by_key[key].get("verdict") != item["semantic_verdict"]:
            raise ValueError(f"review verdict mismatch: {key}")

    items = [review_by_key[key] for key in sorted(expected)]
    run_trial_by_key = run_by_key
    total = len(items)
    passed = sum(bool(item["task_pass"]) for item in items)
    mechanical = sum(item["mechanical_status"] == "needs_review" for item in items)
    semantic = sum(item["semantic_verdict"] == "pass" for item in items)
    if review.get("task_success") != _ratio(passed, total):
        raise ValueError("review summary does not match reviewed trials")
    latencies = [float(run_trial_by_key[key]["latency_ms"]) for key in sorted(expected)]
    sorted_latencies = sorted(latencies)
    p95 = sorted_latencies[math.ceil(0.95 * len(sorted_latencies)) - 1]
    usage_rows = [run_trial_by_key[key].get("usage", {}) for key in sorted(expected)]
    prompt_tokens = _total([row.get("prompt_tokens") for row in usage_rows])
    completion_tokens = _total([row.get("completion_tokens") for row in usage_rows])
    raw_responses = progress.get("raw_responses", {})
    usage_by_case = progress.get("usage_by_case", {})
    unaccounted = progress.get("unaccounted_attempts_by_case", {})
    review_errors = progress.get("review_errors", {})
    if not all(
        isinstance(value, dict)
        for value in (raw_responses, usage_by_case, unaccounted, review_errors)
    ):
        raise ValueError("judge progress accounting must be objects")
    if any(not isinstance(row, dict) for row in usage_by_case.values()):
        raise ValueError("judge usage entries must be objects")
    if any(not isinstance(value, str) for value in raw_responses.values()):
        raise ValueError("judge raw responses must be text")
    if any(case_id not in case_ids for case_id in raw_responses) or any(
        case_id not in case_ids for case_id in usage_by_case
    ):
        raise ValueError("judge progress references an unknown case")
    if any(
        case_id not in case_ids or type(count) is not int or count < 0
        for case_id, count in unaccounted.items()
    ):
        raise ValueError("invalid unaccounted reviewer attempt count")
    if any(case_id not in case_ids for case_id in review_errors):
        raise ValueError("review error references an unknown case")
    for case_id, error in review_errors.items():
        if (
            not isinstance(error, dict)
            or error.get("error_type") != "invalid_judge_output"
            or case_id not in raw_responses
            or error.get("response_sha256")
            != hashlib.sha256(raw_responses[case_id].encode("utf-8")).hexdigest()
        ):
            raise ValueError("judge protocol error is not bound to its raw response")
        if any(
            decision_by_key[(case_id, trial_index)].get("verdict") != "fail"
            for trial_index in range(1, repetitions + 1)
        ):
            raise ValueError("invalid judge output was not failed closed")
    unaccounted_count = sum(unaccounted.values())
    judge_usage_complete = not unaccounted_count and set(raw_responses) == set(
        usage_by_case
    )
    judge_usage_rows = list(usage_by_case.values())
    judge_prompt_tokens = (
        _total([row.get("prompt_tokens") for row in judge_usage_rows])
        if judge_usage_complete
        else None
    )
    judge_completion_tokens = (
        _total([row.get("completion_tokens") for row in judge_usage_rows])
        if judge_usage_complete
        else None
    )
    run_bytes = run_path.read_bytes()
    review_bytes = review_path.read_bytes()
    trace_set_hash = hashlib.sha256(
        "\n".join(sorted(trace_hashes)).encode()
    ).hexdigest()

    def metric(name: str) -> dict[str, int]:
        return _ratio(
            sum(bool(run_by_key[key].get(name)) for key in sorted(expected)), total
        )

    per_case: dict[str, dict[str, int]] = {}
    for case_id in case_ids:
        case_items = [item for item in items if item["case_id"] == case_id]
        per_case[case_id] = _ratio(
            sum(bool(item["task_pass"]) for item in case_items), len(case_items)
        )

    return {
        "run_id": run_root.name,
        "provider": run["provider"],
        "model": run["model"],
        "started_at_utc": run.get("started_at_utc"),
        "source_revision": (run.get("provenance") or {}).get("source_revision"),
        "runtime_config": runtime_config,
        "run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "judge_progress_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
        "trace_set_sha256": trace_set_hash,
        "review_source_sha256": review_source,
        "judge_protocol_version": judge_protocol_version,
        "scenario_count": len(case_ids),
        "repetitions_per_scenario": repetitions,
        "trial_count": total,
        "task_success": _ratio(passed, total),
        "mechanical_pass": _ratio(mechanical, total),
        "semantic_pass": _ratio(semantic, total),
        "tool_selection": metric("tool_selection"),
        "argument_match": metric("argument_match"),
        "final_state": metric("final_state"),
        "error_count": sum(
            run_by_key[key].get("error") is not None for key in sorted(expected)
        ),
        "unresolved_case_ids": sorted(
            case_id
            for case_id, stats in per_case.items()
            if stats["passed"] < stats["total"]
        ),
        "per_case_task_success": per_case,
        "per_split_task_success": _bucket_metrics(items),
        "latency_ms": {
            "p50": statistics.median(latencies),
            "p95": p95,
            "min": min(latencies),
            "max": max(latencies),
        },
        "request_count": sum(
            int(run_by_key[key].get("request_count", 0)) for key in sorted(expected)
        ),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "judge_request_count": len(raw_responses) + unaccounted_count,
        "review_protocol_error_count": len(review_errors),
        "judge_usage": {
            "prompt_tokens": judge_prompt_tokens,
            "completion_tokens": judge_completion_tokens,
        },
        "judge_model": progress.get("judge_model"),
        "reviewer": review.get("reviewer"),
    }

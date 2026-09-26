"""Validate and summarize comparable, hash-frozen Agent regression runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from modules.vehicle_ai.evaluation.regression_html import render_html


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
        name: _ratio(sum(item["task_pass"] for item in values), len(values))
        for name, values in sorted(buckets.items())
    }


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
        if (
            trace_data.get("case", {}).get("id") != key[0]
            or trace_trial.get("case_id") != key[0]
            or trace_trial.get("trial_index") != key[1]
            or trace_trial.get("provider") != run.get("provider")
            or trace_trial.get("model") != run.get("model")
            or trace_trial.get("case_sha256") != item.get("case_sha256")
        ):
            raise ValueError(f"trace identity mismatch: {key}")
        run_by_key[key] = item
        trace_hashes.append(trace_hash)
    if set(run_by_key) != expected:
        raise ValueError("run is missing a frozen case or repetition")

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
    judge_usage_rows = list(progress.get("usage_by_case", {}).values())
    prompt_tokens = _total([row.get("prompt_tokens") for row in usage_rows])
    completion_tokens = _total([row.get("completion_tokens") for row in usage_rows])
    judge_prompt_tokens = _total([row.get("prompt_tokens") for row in judge_usage_rows])
    judge_completion_tokens = _total(
        [row.get("completion_tokens") for row in judge_usage_rows]
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
        "run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
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
        "judge_request_count": len(progress.get("raw_responses", {})),
        "judge_usage": {
            "prompt_tokens": judge_prompt_tokens,
            "completion_tokens": judge_completion_tokens,
        },
        "judge_model": progress.get("judge_model"),
        "reviewer": review.get("reviewer"),
    }


def _cost_cny(
    usage: dict[str, int | None], rates: dict[str, Any] | None
) -> float | None:
    if rates is None:
        return None
    input_rate = rates.get("input_cny_per_million")
    output_rate = rates.get("output_cny_per_million")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        not isinstance(input_rate, (int, float))
        or not isinstance(output_rate, (int, float))
        or type(prompt) is not int
        or type(completion) is not int
    ):
        return None
    return (prompt * input_rate + completion * output_rate) / 1_000_000


def build_comparison(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    stage_evidence: list[dict[str, str]],
    pricing: dict[str, Any],
) -> dict[str, Any]:
    if set(before) != {"qwen", "glm"} or set(after) != {"qwen", "glm"}:
        raise ValueError("comparison requires qwen and glm before/after batches")
    providers: dict[str, Any] = {}
    rates = pricing.get("providers", {})
    for provider in ("qwen", "glm"):
        old, new = before[provider], after[provider]
        if (old["provider"], new["provider"]) != (provider, provider):
            raise ValueError(f"provider mapping mismatch: {provider}")
        if (
            old["scenario_count"] != new["scenario_count"]
            or old["per_case_task_success"].keys()
            != new["per_case_task_success"].keys()
        ):
            raise ValueError(f"scenario denominator changed: {provider}")
        if old["judge_protocol_version"] != new["judge_protocol_version"]:
            raise ValueError(f"judge protocol changed: {provider}")
        rate = rates.get(provider)
        if rate is not None and rate.get("model") not in {old["model"], new["model"]}:
            rate = None
        judge_rate = rates.get("qwen")
        expected_judge_model = (
            f"qwen/{judge_rate['model']}" if judge_rate is not None else None
        )
        providers[provider] = {
            "before": old,
            "after": new,
            "task_success_delta_percentage_points": 100
            * (
                new["task_success"]["passed"] / new["task_success"]["total"]
                - old["task_success"]["passed"] / old["task_success"]["total"]
            ),
            "before_agent_cost_estimate_cny": _cost_cny(old["usage"], rate),
            "after_agent_cost_estimate_cny": _cost_cny(new["usage"], rate),
            "after_judge_cost_estimate_cny": _cost_cny(
                new["judge_usage"],
                judge_rate if new.get("judge_model") == expected_judge_model else None,
            ),
            "before_judge_cost_estimate_cny": _cost_cny(
                old["judge_usage"],
                judge_rate if old.get("judge_model") == expected_judge_model else None,
            ),
        }
    return {
        "title": "A6 A5 后在线 Agent 回归对照",
        "pricing_snapshot_date": pricing.get("captured_at"),
        "pricing_note": pricing.get("note", ""),
        "pricing_sources": [
            {
                "provider": provider,
                "model": item.get("model"),
                "source_url": item.get("source_url"),
                "note": item.get("note", ""),
            }
            for provider, item in sorted(rates.items())
        ],
        "providers": providers,
        "stage_evidence": stage_evidence,
        "limitations": [
            "40 个冻结场景由项目内部 AI 自审，非独立人工金标准；每场景重复 3 次不增加独立样本量。",
            "语义复核由同一 Qwen 审核器按 protocol v4 辅助完成，存在同源偏差，不等同人工审定。",
            "模型名称是服务别名；提供商未提供可核对的权重/服务版本摘要，跨日期差异不能完全归因于代码变更。",
            "Agent 运行只用合成场景与模拟工具；不代表真实车辆、道路安全或感知准确率。",
            "Token 费用为指定时点公开标准价重算的估算，非账户账单；不含缓存折扣、免费额度、促销或审核失败重试等无法验证的计费调整。",
            "GLM 当前 BigModel endpoint 的精确官方计费档未在本报告核验，相关费用为 N/A。",
        ],
    }


def _format_ratio(value: dict[str, int]) -> str:
    return (
        f"{value['passed']}/{value['total']} ({value['passed'] / value['total']:.1%})"
    )


def _format_cost(value: float | None) -> str:
    return "N/A" if value is None else f"¥{value:.4f}"


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        f"# {comparison['title']}",
        "",
        "> 同一 40 条冻结场景、当前 rubric 与 AI 语义复核协议 v4；Qwen/GLM 各 3 次。重复 trial 不是独立样本。指标不合成为综合分。",
        "",
        "## 前后回归",
        "",
        "| 模型 | 版本 | Task Success（任务成功） | Mechanical Pass（机械通过） | Tool / Argument / State（工具/参数/状态匹配） | 异常 | 延迟 p50/p95 | 请求 | 输入/输出 token | Agent 费用估算 | Qwen 审核费用估算 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in ("qwen", "glm"):
        item = comparison["providers"][provider]
        for label, key in (("优化前", "before"), ("A5 后", "after")):
            data = item[key]
            prefix = "before" if key == "before" else "after"
            usage = data["usage"]
            latency = data["latency_ms"]
            lines.append(
                f"| {provider.upper()} | {label} | {_format_ratio(data['task_success'])} | "
                f"{_format_ratio(data['mechanical_pass'])} | "
                f"{_format_ratio(data['tool_selection'])} / {_format_ratio(data['argument_match'])} / {_format_ratio(data['final_state'])} | "
                f"{data['error_count']}/{data['trial_count']} | {latency['p50']:.0f}/{latency['p95']:.0f} ms | "
                f"{data['request_count']} | {usage['prompt_tokens'] if usage['prompt_tokens'] is not None else 'N/A'} / {usage['completion_tokens'] if usage['completion_tokens'] is not None else 'N/A'} | "
                f"{_format_cost(item[f'{prefix}_agent_cost_estimate_cny'])} | "
                f"{_format_cost(item[f'{prefix}_judge_cost_estimate_cny'])} |"
            )
        delta = item["task_success_delta_percentage_points"]
        lines.append(
            f"| 变化 | {provider.upper()} | {delta:+.1f} 个百分点 | | | | | | | | |"
        )
    lines.extend(
        [
            "",
            "每行 Task Success 分母均为 120 个 trial（40 场景 × 3 次）；相同场景的重复性另由逐场景结果体现。Mechanical Pass 是确定性规则通过比例；Task Success 还要求 v4 AI 语义审查通过。`needs_review` 不计为完成任务成功。",
            "",
            "## 阶段证据（分母彼此独立）",
            "",
            "| 阶段 | 指标 | 结果 | 证据口径 |",
            "|---|---|---:|---|",
        ]
    )
    for row in comparison["stage_evidence"]:
        lines.append(
            f"| {row['stage']} | {row['name']} | {row['value']} | {row['scope']} |"
        )
    lines.extend(["", "## 未通过场景", ""])
    for provider in ("qwen", "glm"):
        item = comparison["providers"][provider]["after"]
        ids = item["unresolved_case_ids"]
        rendered = (
            "、".join(
                f"[{case_id}](../../scenarios/agent_eval/golden/cases/{case_id}.yaml)"
                for case_id in ids
            )
            or "无"
        )
        lines.append(f"- {provider.upper()}：{rendered}")
    lines.extend(["", "## 费用与复现来源", ""])
    lines.append(
        f"公开目录价格快照：{comparison.get('pricing_snapshot_date') or '未提供'}。{comparison['pricing_note']}"
    )
    for source in comparison.get("pricing_sources", []):
        url = source.get("source_url")
        suffix = (
            f"[{url}]({url})"
            if isinstance(url, str) and url.startswith("https://")
            else "链接未核验"
        )
        lines.append(
            f"- {source['provider'].upper()} {source.get('model') or ''} 官方来源：{suffix}；{source.get('note', '')}"
        )
    for provider in ("qwen", "glm"):
        lines.append(f"- {provider.upper()}：")
        for key in ("before", "after"):
            data = comparison["providers"][provider][key]
            lines.append(
                f"  - {key} run `{data['run_id']}`；revision `{data.get('source_revision') or '历史来源未随 run 固化'}`；"
                f"run SHA-256 `{data['run_sha256']}`；review SHA-256 `{data['review_sha256']}`；"
                f"trace 集 SHA-256 `{data['trace_set_sha256']}`；语义 protocol v{data['judge_protocol_version']}。"
            )
    lines.extend(["", "## 限制", ""])
    lines.extend(f"- {item}" for item in comparison["limitations"])
    lines.extend(
        [
            "",
            "在线请求只包含版本化合成场景、当前工具协议及合成 Agent 回复；本报告不包含舱内/舱外视频、个人数据、密钥或逐请求原始 trace。完整 trace 和 AI 判定原文保留在本机 Git 忽略目录，按 run ID 和 SHA-256 核对。",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(comparison: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_markdown(comparison), encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(comparison), encoding="utf-8")

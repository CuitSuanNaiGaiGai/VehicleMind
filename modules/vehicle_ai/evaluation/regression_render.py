"""Build user-facing A6 comparison and report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.vehicle_ai.evaluation.regression_html import render_html
from modules.vehicle_ai.evaluation.regression_subsets import render_subset_markdown


def _matching_rate(
    provider: str, model: str, rates: dict[str, Any]
) -> dict[str, Any] | None:
    rate = rates.get(provider)
    return rate if isinstance(rate, dict) and rate.get("model") == model else None


def _format_runtime_config(config: dict[str, Any]) -> str:
    labels = {
        "temperature": "温度",
        "timeout_seconds": "模型请求超时秒数",
        "max_tool_rounds": "工具轮次上限",
        "turn_timeout_seconds": "单轮超时秒数",
        "max_tool_calls": "工具调用上限",
        "max_task_trace_events": "Agent 轨迹事件上限",
    }
    return "，".join(
        f"{labels[key]}={config[key] if config[key] is not None else '未记录'}"
        for key in labels
    )


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
        if old["model"] != new["model"]:
            raise ValueError(f"model changed across comparison: {provider}")
        old_config, new_config = old.get("runtime_config"), new.get("runtime_config")
        if not isinstance(old_config, dict) or not isinstance(new_config, dict):
            raise ValueError(f"runtime budget is unavailable: {provider}")
        config_fields = (
            "temperature",
            "timeout_seconds",
            "max_tool_rounds",
            "turn_timeout_seconds",
            "max_tool_calls",
            "max_task_trace_events",
        )
        if any(
            old_config.get(field) is not None
            and new_config.get(field) is not None
            and old_config[field] != new_config[field]
            for field in config_fields
        ):
            raise ValueError(f"runtime budget changed across comparison: {provider}")
        unknown_budget_fields = [
            field
            for field in config_fields
            if old_config.get(field) is None or new_config.get(field) is None
        ]
        if (
            old["scenario_count"] != new["scenario_count"]
            or old["per_case_task_success"].keys()
            != new["per_case_task_success"].keys()
        ):
            raise ValueError(f"scenario denominator changed: {provider}")
        if old["judge_protocol_version"] != new["judge_protocol_version"]:
            raise ValueError(f"judge protocol changed: {provider}")
        before_rate = _matching_rate(provider, old["model"], rates)
        after_rate = _matching_rate(provider, new["model"], rates)
        judge_rate = rates.get("qwen")
        expected_judge_model = (
            f"qwen/{judge_rate['model']}" if judge_rate is not None else None
        )
        providers[provider] = {
            "before": old,
            "after": new,
            "task_success_delta_percentage_points": (
                100
                * (
                    new["task_success"]["passed"] / new["task_success"]["total"]
                    - old["task_success"]["passed"] / old["task_success"]["total"]
                )
                if not unknown_budget_fields
                else None
            ),
            "comparison_status": (
                "budget_verified"
                if not unknown_budget_fields
                else "descriptive_only_missing_historical_budget"
            ),
            "unknown_budget_fields": unknown_budget_fields,
            "before_agent_cost_estimate_cny": _cost_cny(old["usage"], before_rate),
            "after_agent_cost_estimate_cny": _cost_cny(new["usage"], after_rate),
            "after_judge_cost_estimate_cny": _cost_cny(
                new["judge_usage"],
                judge_rate if new.get("judge_model") == expected_judge_model else None,
            ),
            "before_judge_cost_estimate_cny": _cost_cny(
                old["judge_usage"],
                judge_rate if old.get("judge_model") == expected_judge_model else None,
            ),
        }
    limitations = [
        "40 个冻结场景由项目内部 AI 自审，非独立人工金标准；每场景重复 3 次不增加独立样本量。",
        "语义复核由同一 Qwen 审核器按 protocol v4 辅助完成，存在同源偏差，不等同人工审定。",
        "模型名称是服务别名；提供商未提供可核对的权重/服务版本摘要，跨日期差异不能完全归因于代码变更。",
        "Agent 运行只用合成场景与模拟工具；不代表真实车辆、道路安全或感知准确率。",
        "Token 费用为指定时点公开标准价重算的估算，非账户账单；不含缓存折扣、免费额度、促销或审核失败重试等无法验证的计费调整。",
        "GLM 当前 BigModel endpoint 的精确官方计费档未在本报告核验，相关费用为 N/A。",
    ]
    if any(
        pair["comparison_status"] != "budget_verified" for pair in providers.values()
    ):
        limitations.append(
            "历史基线没有记录全部 Agent 运行预算；虽然已记录参数没有发现差异，但 Task Success 差值标为 N/A，只并列展示两次观察结果，不将变化归因于 A5 代码。"
        )
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
        "limitations": limitations,
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
        "| 模型 | 对照 | Task Success（任务成功） | Mechanical Pass（机械通过） | Semantic Pass（语义通过） | Tool / Argument / State（工具/参数/状态匹配） | Agent 异常 | 审核格式错误 | 延迟 p50/p95 | Agent 请求 | 审核请求 | Agent 输入/输出 token | Agent 费用估算 | Qwen 审核费用估算 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in ("qwen", "glm"):
        item = comparison["providers"][provider]
        for label, key in (("优化前", "before"), ("A5 后", "after")):
            data = item[key]
            prefix = "before" if key == "before" else "after"
            usage = data["usage"]
            latency = data["latency_ms"]
            lines.append(
                f"| {provider.upper()} / {data['model']} | {label} | {_format_ratio(data['task_success'])} | "
                f"{_format_ratio(data['mechanical_pass'])} | "
                f"{_format_ratio(data['semantic_pass'])} | "
                f"{_format_ratio(data['tool_selection'])} / {_format_ratio(data['argument_match'])} / {_format_ratio(data['final_state'])} | "
                f"{data['error_count']}/{data['trial_count']} | {data['review_protocol_error_count']} | "
                f"{latency['p50']:.0f}/{latency['p95']:.0f} ms | "
                f"{data['request_count']} | {data['judge_request_count']} | "
                f"{usage['prompt_tokens'] if usage['prompt_tokens'] is not None else 'N/A'} / {usage['completion_tokens'] if usage['completion_tokens'] is not None else 'N/A'} | "
                f"{_format_cost(item[f'{prefix}_agent_cost_estimate_cny'])} | "
                f"{_format_cost(item[f'{prefix}_judge_cost_estimate_cny'])} |"
            )
        delta = item["task_success_delta_percentage_points"]
        delta_text = (
            f"{delta:+.1f} 个百分点"
            if delta is not None
            else "N/A（历史预算字段未记录，仅作结果并列）"
        )
        status = (
            "预算已核验"
            if item["comparison_status"] == "budget_verified"
            else "仅并列观察"
        )
        lines.append(
            f"| 变化（{status}） | {provider.upper()} | {delta_text} | | | | | | | | | | | |"
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
    lines.append(render_subset_markdown(comparison))
    lines.extend(["", "## 未通过场景", ""])
    for provider in ("qwen", "glm"):
        item = comparison["providers"][provider]["after"]
        ids = item["unresolved_case_ids"]
        rendered = (
            "、".join(
                f"[{case_id}](../../../scenarios/agent_eval/golden/cases/{case_id}.yaml)"
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
            if data.get("runtime_config"):
                config = data["runtime_config"]
                lines.append("  - 运行配置：" + _format_runtime_config(config) + ".")
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

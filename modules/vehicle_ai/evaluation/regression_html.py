"""Compact Chinese HTML rendering for A6 regression summaries."""

from __future__ import annotations

from html import escape
from typing import Any


def _rate(value: dict[str, int]) -> str:
    percent = value["passed"] / value["total"] if value["total"] else 0.0
    return f"{value['passed']}/{value['total']} · {percent:.1%}"


def _money(value: float | None) -> str:
    return "N/A" if value is None else f"¥{value:.4f}"


def _delta(pair: dict[str, Any]) -> str:
    value = pair["task_success_delta_percentage_points"]
    return "N/A" if value is None else f"{value:+.1f}"


def render_html(comparison: dict[str, Any]) -> str:
    rows: list[str] = []
    for provider in ("qwen", "glm"):
        pair = comparison["providers"][provider]
        for label, key, cost_key in (
            ("优化前", "before", "before"),
            ("A5 后", "after", "after"),
        ):
            data = pair[key]
            latency = data["latency_ms"]
            usage = data["usage"]
            rows.append(
                "<tr>"
                f"<th>{escape(provider.upper())} / {escape(str(data.get('model') or '模型未记录'))}</th><td>{label}</td>"
                f"<td>{_rate(data['task_success'])}</td>"
                f"<td>{_rate(data['mechanical_pass'])}</td>"
                f"<td>{_rate(data['semantic_pass'])}</td>"
                f"<td>{_rate(data['tool_selection'])} / {_rate(data['argument_match'])} / {_rate(data['final_state'])}</td>"
                f"<td>{data['error_count']}/{data['trial_count']}</td>"
                f"<td>{data.get('review_protocol_error_count', 0)}</td>"
                f"<td>{latency['p50']:.0f} / {latency['p95']:.0f} ms</td>"
                f"<td>{data['request_count']}</td>"
                f"<td>{data.get('judge_request_count', 'N/A')}</td>"
                f"<td>{usage['prompt_tokens'] if usage['prompt_tokens'] is not None else 'N/A'} / {usage['completion_tokens'] if usage['completion_tokens'] is not None else 'N/A'}</td>"
                f"<td>{_money(pair[f'{cost_key}_agent_cost_estimate_cny'])}</td>"
                f"<td>{_money(pair[f'{cost_key}_judge_cost_estimate_cny'])}</td>"
                "</tr>"
            )

    stages = "".join(
        "<tr>"
        f"<th>{escape(item['stage'])}</th><td>{escape(item['name'])}</td>"
        f"<td>{escape(item['value'])}</td><td>{escape(item['scope'])}</td>"
        "</tr>"
        for item in comparison["stage_evidence"]
    )
    deltas = "".join(
        "<article class='delta'>"
        f"<h2>{escape(provider.upper())} Task Success（任务成功）变化</h2>"
        f"<p>{_delta(pair)}"
        f"<small>{' 个百分点' if pair['task_success_delta_percentage_points'] is not None else ''}</small></p>"
        f"<p class='muted'>{'前后预算字段完整且一致。' if pair.get('comparison_status') == 'budget_verified' else '历史预算字段未记录，仅作结果并列，不作变化归因。'}</p>"
        "</article>"
        for provider, pair in comparison["providers"].items()
    )
    limitations = "".join(
        f"<li>{escape(item)}</li>" for item in comparison["limitations"]
    )
    price_sources = []
    for item in comparison.get("pricing_sources", []):
        url = item.get("source_url")
        if isinstance(url, str) and url.startswith("https://"):
            source = f"<a href='{escape(url, quote=True)}'>{escape(url)}</a>"
        else:
            source = "来源未核验"
        price_sources.append(
            f"<li>{escape(item['provider'].upper())} {escape(item.get('model') or '')}：{source}。{escape(item.get('note', ''))}</li>"
        )
    failure_sections: list[str] = []
    run_evidence: list[str] = []
    for provider in ("qwen", "glm"):
        pair = comparison["providers"][provider]
        ids = pair["after"]["unresolved_case_ids"]
        if ids:
            links = "、".join(
                f"<a href='../../../scenarios/agent_eval/golden/cases/{escape(case_id)}.yaml'>{escape(case_id)}</a>"
                for case_id in ids
            )
        else:
            links = "无"
        failure_sections.append(
            f"<li><strong>{escape(provider.upper())}：</strong>{links}</li>"
        )
        for label, key in (("优化前", "before"), ("A5 后", "after")):
            data = pair[key]
            config = data.get("runtime_config") or {}
            config_labels = {
                "temperature": "温度",
                "timeout_seconds": "模型请求超时",
                "max_tool_rounds": "工具轮次上限",
                "turn_timeout_seconds": "单轮超时",
                "max_tool_calls": "工具调用上限",
                "max_task_trace_events": "Agent 轨迹事件上限",
            }
            rendered_config = "；".join(
                f"{config_labels[field]}={config.get(field) if config.get(field) is not None else '未记录'}"
                for field in config_labels
            )
            run_evidence.append(
                f"<li>{escape(provider.upper())} / {escape(str(data['model']))} · {label}："
                f"run <code>{escape(str(data['run_id']))}</code>；"
                f"运行预算：{escape(rendered_config or '未记录')}；"
                f"revision <code>{escape(str(data.get('source_revision') or 'N/A'))}</code>；"
                f"run/review/judge/trace SHA-256 前 12 位："
                f"<code>{escape(str(data['run_sha256'])[:12])}</code> / "
                f"<code>{escape(str(data['review_sha256'])[:12])}</code> / "
                f"<code>{escape(str(data.get('judge_progress_sha256') or 'N/A'))[:12]}</code> / "
                f"<code>{escape(str(data['trace_set_sha256'])[:12])}</code></li>"
            )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(comparison["title"])}</title><style>
:root{{--ink:#172033;--muted:#60708a;--line:#e1e7ef;--blue:#315efb;--paper:#f5f7fb;--card:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1200px;margin:38px auto;padding:0 22px}}header,.panel{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px;margin-bottom:18px;box-shadow:0 8px 28px #233b6310}}
.eyebrow{{color:var(--blue);font-weight:700;letter-spacing:.08em}}h1{{font-size:30px;margin:6px 0}}h2{{font-size:20px;margin:0 0 12px}}p{{margin:8px 0}}.muted{{color:var(--muted)}}.deltas{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:18px}}
.delta{{background:#edf2ff;border-radius:14px;padding:18px 22px}}.delta p{{font-size:34px;color:var(--blue);font-weight:750;margin:0}}small{{font-size:14px;font-weight:500}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:940px}}th,td{{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}}thead th{{background:#f8faff;position:sticky;top:0}}ul{{padding-left:22px}}.tag{{display:inline-block;border-radius:99px;background:#eaf0ff;color:#2546bb;padding:3px 10px;margin-right:6px}}
@media(max-width:680px){{main{{margin:16px auto;padding:0 12px}}header,.panel{{padding:18px}}h1{{font-size:24px}}.deltas{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">VEHICLEMIND · A6 回归证据</div><h1>{escape(comparison["title"])}</h1>
<p>同一 40 条冻结场景；每模型每场景重复 3 次。机械检查与 AI 语义复核分开显示，不合成总分。</p>
<p class="muted">Task Success（任务成功）要求机械通过且 protocol v4 语义审查通过。AI 辅助审核不是独立人工金标准。</p></header>
<section class="deltas">{deltas}</section>
<section class="panel"><h2>在线 Agent 前后对照</h2><div class="scroll"><table><thead><tr><th>模型</th><th>对照</th><th>Task Success（任务成功）</th><th>Mechanical Pass（机械通过）</th><th>Semantic Pass（语义通过）</th><th>Tool / Argument / State（工具/参数/状态匹配）</th><th>Agent 异常</th><th>审核格式错误</th><th>p50 / p95 延迟</th><th>Agent 请求数</th><th>审核请求数</th><th>Agent 输入 / 输出 token</th><th>Agent 费用估算</th><th>Qwen 审核费用估算</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">每个核心率的分母是 120 个 trial（40 场景 × 3 次）；重复次数不代表独立样本量。输入/输出 token 缺报时为 N/A。</p></section>
<section class="panel"><h2>A1–A5 阶段证据（各用各自分母）</h2><div class="scroll"><table><thead><tr><th>阶段</th><th>指标</th><th>结果</th><th>口径</th></tr></thead><tbody>{stages}</tbody></table></div></section>
<section class="panel"><h2>A5 后未通过场景</h2><ul>{"".join(failure_sections)}</ul><p class="muted">链接只指向公开冻结 case，不展示 prompt 全文或原始模型 trace。</p></section>
<section class="panel"><h2>运行证据与复现配置</h2><ul>{"".join(run_evidence)}</ul><p class="muted">优化前与 A5 后的实际模型名、模型运行预算及来源短哈希均列于此；无原始 prompt 或逐条回答。</p></section>
<section class="panel"><h2>证据与限制</h2><p>价格快照：{escape(str(comparison.get("pricing_snapshot_date") or "N/A"))}。{escape(comparison.get("pricing_note", ""))}</p><ul>{"".join(price_sources)}</ul><ul>{limitations}</ul>
<p class="muted">原始运行与逐条审核保存在本机 Git 忽略的 runs/；本 HTML 仅包含汇总。</p></section>
</main></body></html>"""

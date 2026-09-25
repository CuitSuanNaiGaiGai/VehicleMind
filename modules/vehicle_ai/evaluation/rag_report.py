"""Chinese, evidence-first HTML report for opt-in knowledge trials."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence

from modules.vehicle_ai.evaluation.rag_runner import RagTrial


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _metric(label: str, data: Mapping[str, object] | None) -> str:
    value_number = data.get("value") if data else None
    if not isinstance(value_number, (int, float)):
        value = "N/A（未评测）"
    else:
        value = f"{value_number * 100:.1f}%"
        assert data is not None
        value += f" <small>{_escape(data.get('numerator'))}/{_escape(data.get('denominator'))}</small>"
    return f"<div class='metric'><span>{_escape(label)}</span><strong>{value}</strong></div>"


def _trial_card(trial: RagTrial) -> str:
    if trial.error:
        status = f"<span class='badge error'>失败：{_escape(trial.error)}</span>"
    elif trial.answer:
        status = "<span class='badge'>已完成</span>"
    else:
        status = "<span class='badge muted'>无回答</span>"
    evidence = ""
    visible_retrieval = trial.agent_retrieval or trial.retrieval
    if visible_retrieval is not None:
        for chunk in visible_retrieval.chunks:
            ref = chunk.reference
            evidence += (
                "<div class='source'><div class='source-head'>"
                f"<b>[{_escape(chunk.source_id)}] {_escape(ref.title)}</b>"
                f"<span>{_escape(ref.section)}</span></div>"
                f"<p>{_escape(chunk.text)}</p>"
                f"<a href='{_escape(ref.source_uri)}' target='_blank' rel='noopener noreferrer'>查看证据来源 ↗</a></div>"
            )
    if not evidence:
        evidence = "<p class='muted'>本次未返回可引用证据。</p>"
    tokens = (
        ", ".join(f"{key}: {value}" for key, value in trial.agent_usage.items())
        if trial.agent_usage
        else "N/A（服务未返回）"
    )
    live_state = (
        "；".join(
            f"{_escape(key)}={_escape(value)}"
            for key, value in trial.live_context.items()
        )
        if trial.live_context
        else "本题未注入已验证的实时车况"
    )
    raw = json.dumps(
        {
            "case_id": trial.case_id,
            "expected_source_ids": trial.expected_source_ids,
            "retrieved_source_ids": trial.retrieved_source_ids,
            "request_id": visible_retrieval.request_id if visible_retrieval else None,
            "agent_tool_request_id": trial.agent_retrieval.request_id
            if trial.agent_retrieval
            else None,
            "error": trial.error,
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"<article class='trial'><div class='trial-top'><span>{_escape(trial.case_id)} · {_escape(trial.profile)}</span>{status}</div>"
        f"<h2>{_escape(trial.query)}</h2>"
        "<div class='answer-label'>Agent 最终回答</div>"
        f"<div class='answer'>{_escape(trial.answer) if trial.answer else '无回答'}</div>"
        "<div class='facts'>"
        f"<span>知识工具调用：{'是' if trial.knowledge_tool_called else '否'}</span>"
        f"<span>检索耗时：{_escape(visible_retrieval.latency_ms if visible_retrieval else 'N/A')} ms</span>"
        f"<span>试次总耗时：{_escape(trial.elapsed_ms)} ms</span>"
        f"<span>Agent tokens：{_escape(tokens)}</span>"
        f"</div><div class='meta'>实时状态来源：{live_state}</div><h3>检索证据</h3>"
        f"{evidence}<details><summary>查看机器记录</summary><pre>{_escape(raw)}</pre></details></article>"
    )


def render_report(
    trials: Sequence[RagTrial],
    summary: Mapping[str, object],
    metadata: Mapping[str, object],
) -> str:
    recall = summary.get("recall_at_5")
    support = summary.get("citation_support")
    abstention = summary.get("abstention")
    abstention = abstention if isinstance(abstention, dict) else {}
    metrics = "".join(
        (
            _metric(
                "Recall@5（前五条证据召回率）",
                recall if isinstance(recall, dict) else None,
            ),
            _metric(
                "Citation Support（引用支持率）",
                support if isinstance(support, dict) else None,
            ),
            _metric(
                "No-answer Abstention（AI 审查无答案弃答率）",
                abstention.get("no_answer"),
            ),
            _metric(
                "Scope Abstention（AI 审查范围弃答率）",
                abstention.get("profile_mismatch"),
            ),
            f"<div class='metric'><span>Profile Leakage（检索来源范围泄漏）</span><strong>{_escape(summary.get('profile_leakage', 'N/A'))}</strong></div>",
        )
    )
    meta = " · ".join(
        f"{_escape(key)}: {_escape(value)}" for key, value in metadata.items()
    )
    review_status = _escape(summary.get("citation_review_status", "not_reviewed"))
    review_errors = summary.get("citation_review_errors")
    review_error_count = len(review_errors) if isinstance(review_errors, dict) else 0
    abstention_status = _escape(summary.get("abstention_review_status", "not_reviewed"))
    abstention_errors = summary.get("abstention_review_errors")
    abstention_error_count = (
        len(abstention_errors) if isinstance(abstention_errors, dict) else 0
    )
    case_count = _escape(summary.get("case_count", len(trials)))
    cards = "".join(_trial_card(trial) for trial in trials)
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>VehicleMind · 知识增强 Agent 评测</title>
<style>
:root{{--bg:#f4f7fb;--ink:#142334;--sub:#526376;--line:#dce5ef;--blue:#2166c6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}}
header{{background:#102d50;color:white;padding:40px max(24px,calc((100vw - 1120px)/2))}}
header h1{{font-size:32px;margin:0 0 10px}}header p{{margin:0;color:#d8e9ff}}
main{{max-width:1168px;margin:auto;padding:28px 24px 60px}}
.notice{{padding:14px 18px;border-left:4px solid #e9a43a;background:#fff8e8;color:#604412;border-radius:8px;margin-bottom:24px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:12px;margin-bottom:30px}}
.metric,.trial{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 18px #173c6510}}
.metric{{padding:16px}}.metric span{{display:block;color:var(--sub);font-size:13px}}.metric strong{{display:block;font-size:24px;margin-top:7px}}small{{font-size:13px;color:var(--sub)}}
.trial{{padding:24px;margin-bottom:20px}}.trial-top{{display:flex;justify-content:space-between;gap:10px;color:var(--sub);font-size:13px}}
.trial h2{{font-size:21px;margin:10px 0 16px}}.trial h3{{margin:22px 0 10px;font-size:16px}}
.badge{{border-radius:99px;background:#e5f4ea;color:#17663a;padding:3px 10px}}.badge.error{{background:#ffe9e9;color:#ae2727}}.badge.muted{{background:#edf0f4;color:#5d6976}}
.answer-label{{font-size:12px;color:var(--sub);margin-bottom:4px}}.answer{{background:#eaf3ff;border-radius:9px;padding:15px 18px;font-size:17px;font-weight:550}}
.facts{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}}.facts span{{font-size:12px;color:#35506d;background:#f0f5fa;padding:5px 9px;border-radius:6px}}
.source{{border-top:1px solid var(--line);padding:14px 0}}.source-head{{display:flex;gap:16px;flex-wrap:wrap}}.source-head span,.muted{{color:var(--sub)}}.source p{{margin:6px 0}}a{{color:var(--blue)}}
details{{margin-top:12px;color:var(--sub)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px;border-radius:8px}}
.meta{{font-size:12px;color:var(--sub);margin-bottom:20px;overflow-wrap:anywhere}}
</style></head><body><header><h1>VehicleMind 知识增强 Agent</h1>
<p>按需检索 · 证据引用 · 适用范围隔离 · 真实 Agent 回答</p></header><main>
<div class='notice'>AI 辅助内部评测：本次 {case_count} 条（冻结集共 30 条），不等于独立人工金标准。未完成事实级审查的指标显示 N/A；LightRAG 内部 token 不做估算。</div>
<div class='meta'>{meta} · 引用审查状态：{review_status} · 引用审查错误数：{review_error_count} · 弃答审查状态：{abstention_status} · 弃答审查错误数：{abstention_error_count}</div><section class='metrics'>{metrics}</section>
<h2>逐题回答与证据</h2>{cards}</main></body></html>"""

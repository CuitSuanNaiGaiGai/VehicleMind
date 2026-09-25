"""Chinese HTML and JSON report for the bounded-recovery acceptance suite."""

from __future__ import annotations

import html
import json
from pathlib import Path

from modules.vehicle_ai.evaluation.recovery import RecoveryReport


METRIC_LABELS = {
    "Recovery Success": "Recovery Success（恢复成功率）",
    "Safe Stop": "Safe Stop（安全停止）",
    "Duplicate Write Violations": "Duplicate Write（重复写操作违规）",
    "Budget Compliance": "Budget Compliance（预算合规率）",
    "Confirmation Compliance": "Confirmation Compliance（确认合规率）",
    "Safety Replay Consistency": "Safety Replay Consistency（安全回放一致率）",
}
PLAN_STATUS_LABELS = {
    "RUNNING": "进行中",
    "COMPLETED": "已完成",
    "FAILED": "失败",
    "CANCELLED": "已取消",
    "STOPPED_NO_RESULT": "无结果并停止",
}


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _metric_cards(report: RecoveryReport) -> str:
    cards = []
    for name, metric in report.metrics.items():
        cards.append(
            "<article class='metric'>"
            f"<span>{_escape(METRIC_LABELS.get(name, name))}</span>"
            f"<strong>{metric.numerator} / {metric.denominator}</strong>"
            f"<small>目标：{_escape(metric.target)}</small>"
            "</article>"
        )
    return "".join(cards)


def _case_rows(report: RecoveryReport) -> str:
    rows = []
    for case in report.cases:
        evidence = json.dumps(case.evidence, ensure_ascii=False, indent=2)
        badge = "通过" if case.passed else "失败"
        rows.append(
            "<tr>"
            f"<td><code>{_escape(case.scenario_id)}</code><br>{_escape(case.title)}</td>"
            f"<td class={'pass' if case.passed else 'fail'}>{badge}</td>"
            f"<td>{_escape(PLAN_STATUS_LABELS.get(case.plan_status, case.plan_status))}<br><small>{_escape(case.terminal_reason or '—')}</small></td>"
            f"<td>{case.step_count} / {case.max_steps}<br>恢复 {case.recovery_count} / {case.max_recoveries}</td>"
            f"<td>{case.confirmed_navigation_writes} 次已确认<br>未确认违规 {case.unconfirmed_sensitive_write_violations}<br>查询 {case.search_attempts} 次</td>"
            f"<td><details><summary>查看步骤证据</summary><pre>{_escape(evidence)}</pre></details></td>"
            "</tr>"
        )
    return "".join(rows)


def render_recovery_html(report: RecoveryReport) -> str:
    status = "通过" if report.passed else "存在失败"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>VehicleMind｜Agent 受限计划与恢复评测</title>
<style>
:root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; color: #17212b; background: #f3f6f8; }}
body {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 60px; }}
h1 {{ margin-bottom: 8px; }} .sub {{ color: #607080; }}
.banner {{ padding: 14px 18px; margin: 22px 0; border-radius: 12px; background: #e5f5ee; }}
.banner.fail {{ background: #fff0ed; }} .metrics {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 12px; }}
.metric {{ background: white; border: 1px solid #dbe3e9; border-radius: 12px; padding: 16px; display: grid; gap: 8px; }}
.metric strong {{ font-size: 26px; }} small {{ color: #607080; }} table {{ width: 100%; border-collapse: collapse; background: white; }}
th,td {{ text-align: left; vertical-align: top; padding: 12px; border-bottom: 1px solid #e2e8ed; }} th {{ background: #edf2f5; }}
.pass {{ color: #137348; font-weight: 700; }} .fail {{ color: #b42318; font-weight: 700; }}
details pre {{ white-space: pre-wrap; max-width: 380px; overflow-wrap: anywhere; }}
.table-wrap {{ overflow-x: auto; border: 1px solid #dbe3e9; border-radius: 12px; margin-top: 16px; }}
</style></head><body>
<h1>Agent 受限计划与失败恢复</h1>
<p class="sub">VehicleMind · 场景集 {_escape(report.suite_id)} · 执行方式：确定性脚本模型</p>
<div class="banner {"fail" if not report.passed else ""}">验收结果：{status} · {len(report.cases)} 个固定场景。模拟地点与工具不连接真实车辆或实时地图。</div>
<section class="metrics">{_metric_cards(report)}</section>
<h2>逐场景核验</h2><div class="table-wrap"><table><thead><tr><th>场景</th><th>断言</th><th>计划状态</th><th>步骤 / 恢复预算</th><th>写操作 / 查询</th><th>可追溯证据</th></tr></thead>
<tbody>{_case_rows(report)}</tbody></table></div>
<h2>口径与限制</h2><ul><li>指标展示实际分子/分母；同一固定场景套件用于软件验收，不作为在线 LLM 语义正确率。</li>
<li>候选地点来自内置模拟目录；故障由脚本注入，不能推断真实导航服务可用性。</li>
<li>敏感写操作只会在显式确认后执行；未知结果会读取状态并停止，不自动重放。</li></ul>
</body></html>"""


def write_recovery_report(
    report: RecoveryReport, output_dir: Path | str
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "summary.json"
    html_path = destination / "report.html"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_recovery_html(report), encoding="utf-8")
    return {"json": json_path, "html": html_path}

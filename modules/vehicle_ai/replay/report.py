from __future__ import annotations

import json
import shutil
import tempfile

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from modules.config.snapshot import render_run_artifacts
from modules.vehicle_ai.replay.models import ReplayScenario
from modules.vehicle_ai.replay.trace import ReplayResult, plain_value


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
UI_ROOT = REPOSITORY_ROOT / "apps/vehicle_ai_demo/replay_ui"


@dataclass(frozen=True)
class ReplayArtifactPaths:
    resolved_config: Path
    run_card: Path
    summary_json: Path
    trace_json: Path
    report_html: Path
    report_css: Path
    report_js: Path
    media: Mapping[str, Path]


def _json_text(value: object, *, compact: bool = False) -> str:
    text = json.dumps(
        plain_value(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
    )
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _summary(
    result: ReplayResult,
    scenario: ReplayScenario,
) -> dict[str, object]:
    return {
        "context_schema_version": result.context_schema_version,
        "scenario_schema_version": scenario.schema_version,
        "scenario_id": result.scenario_id,
        "passed": result.passed,
        "semantic_sha256": result.semantic_sha256,
        "assertions": [plain_value(item.__dict__) for item in result.assertions],
        "final_context": plain_value(result.final_context),
        "event_types": list(result.event_types),
        "successful_tools": list(result.successful_tools),
        "unauthorized_sensitive_executions": (result.unauthorized_sensitive_executions),
        "remaining_scripted_responses": result.remaining_scripted_responses,
        "metrics": plain_value(result.metrics),
    }


def _trace(result: ReplayResult) -> list[dict[str, object]]:
    return [
        {
            "sequence": item.sequence,
            "at_ms": item.at_ms,
            "kind": item.kind,
            "data": plain_value(item.data),
        }
        for item in result.trace
    ]


def _observation_status(result: ReplayResult) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for item in result.trace:
        if item.kind != "context_update":
            continue
        domain = str(item.data["domain"])
        latest[domain] = {
            "source": item.data["source"],
            "confidence": item.data["confidence"],
            "valid": item.data["valid"],
            "at_ms": item.at_ms,
            "freshness": "recorded-replay",
        }
    return latest


def _artifact_paths(output_dir: Path) -> ReplayArtifactPaths:
    return ReplayArtifactPaths(
        resolved_config=output_dir / "resolved_config.yaml",
        run_card=output_dir / "run_card.md",
        summary_json=output_dir / "summary.json",
        trace_json=output_dir / "trace.json",
        report_html=output_dir / "report.html",
        report_css=output_dir / "report.css",
        report_js=output_dir / "report.js",
        media={
            "cabin": output_dir / "media/cabin.gif",
            "road": output_dir / "media/road.gif",
        },
    )


def write_replay_report(
    result: ReplayResult,
    scenario: ReplayScenario,
    run_snapshot: Mapping[str, object],
    output_dir: Path,
) -> ReplayArtifactPaths:
    if output_dir.exists():
        raise FileExistsError(f"replay result directory already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        paths = _artifact_paths(staging)
        rendered = render_run_artifacts(run_snapshot)
        summary = _summary(result, scenario)
        trace = _trace(result)

        paths.resolved_config.write_text(rendered.manifest, encoding="utf-8")
        paths.run_card.write_text(rendered.run_card, encoding="utf-8")
        paths.summary_json.write_text(_json_text(summary) + "\n", encoding="utf-8")
        paths.trace_json.write_text(_json_text(trace) + "\n", encoding="utf-8")

        (staging / "media").mkdir()
        shutil.copy2(scenario.media["cabin"], paths.media["cabin"])
        shutil.copy2(scenario.media["road"], paths.media["road"])
        shutil.copy2(UI_ROOT / "report.css", paths.report_css)
        shutil.copy2(UI_ROOT / "report.js", paths.report_js)

        report_data = {
            "title": scenario.title,
            "description": scenario.description,
            "summary": summary,
            "trace": trace,
            "observations": _observation_status(result),
            "media": {"cabin": "media/cabin.gif", "road": "media/road.gif"},
            "provenance": plain_value(run_snapshot.get("provenance", {})),
            "config_sha256": run_snapshot.get("config_sha256"),
            "limitations": [
                "本次回放使用录制的语义观测，并未重新运行感知模型。",
                "结果证明系统集成流程，不代表感知算法精度。",
                "车辆动作仅为模拟执行，不会控制真实车辆。",
            ],
        }
        template = (UI_ROOT / "report.html").read_text(encoding="utf-8")
        if template.count("__VEHICLEMIND_DATA__") != 1:
            raise ValueError("report template must contain one data placeholder")
        paths.report_html.write_text(
            template.replace(
                "__VEHICLEMIND_DATA__", _json_text(report_data, compact=True)
            ),
            encoding="utf-8",
        )
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return _artifact_paths(output_dir)

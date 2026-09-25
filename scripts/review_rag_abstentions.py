"""Run an AI-assisted semantic review of fixed-denominator abstention cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from modules.vehicle_ai.evaluation.rag_cases import load_cases
from modules.vehicle_ai.evaluation.rag_judge import review_abstention_trials
from modules.vehicle_ai.evaluation.rag_metrics import (
    compute_reviewed_abstention_metrics,
)
from modules.vehicle_ai.evaluation.rag_report import render_report
from modules.vehicle_ai.llm import build_llm_client
from scripts.regrade_rag_run import _read_trial


def review_run(run_dir: Path, provider: str, model: str | None = None) -> dict:
    trial_path = run_dir / "trial.jsonl"
    trial_items = [
        json.loads(line) for line in trial_path.read_text(encoding="utf-8").splitlines()
    ]
    trials = tuple(_read_trial(item) for item in trial_items)
    all_cases, _ = load_cases(run_dir / "eval_cases.yaml")
    trial_ids = {trial.case_id for trial in trials}
    cases = tuple(case for case in all_cases if case.case_id in trial_ids)
    load_dotenv()
    selected_model = model or None
    judge = build_llm_client(
        provider, model=selected_model, timeout_seconds=120, temperature=0
    )
    reviews, errors = review_abstention_trials(judge, trials)
    with (run_dir / "abstention_reviews.jsonl").open("w", encoding="utf-8") as output:
        for review in reviews:
            output.write(json.dumps(asdict(review), ensure_ascii=False) + "\n")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = compute_reviewed_abstention_metrics(cases, reviews)
    summary["explicit_wording_match"] = summary.get("abstention")
    summary["abstention"] = {
        key: {
            "numerator": metric.numerator,
            "denominator": metric.denominator,
            "value": metric.value,
        }
        for key, metric in metrics.items()
    }
    summary["abstention_review_status"] = (
        "partial_ai_assisted" if errors else "ai_assisted_semantic"
    )
    summary["abstention_review_errors"] = errors
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_path = run_dir / "runtime_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["abstention_reviewer"] = {
        "provider": provider,
        "model": model or "configured default",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.html").write_text(
        render_report(trials, summary, metadata), encoding="utf-8"
    )
    return summary["abstention"]


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 辅助审查 RAG 无答案与范围边界")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--provider", choices=("qwen", "glm"), required=True)
    parser.add_argument("--model")
    args = parser.parse_args()
    print(
        json.dumps(
            review_run(args.run_dir, args.provider, args.model),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

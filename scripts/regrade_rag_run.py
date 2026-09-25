"""Refresh explicit abstention counts after a transparent rule revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from modules.vehicle_ai.evaluation.rag_cases import load_cases
from modules.vehicle_ai.evaluation.rag_metrics import compute_abstention_metrics
from modules.vehicle_ai.evaluation.rag_report import render_report
from modules.vehicle_ai.evaluation.rag_runner import RagTrial
from modules.vehicle_ai.knowledge.models import (
    KnowledgeReference,
    RetrievedChunk,
    RetrievalResult,
)
from scripts.run_knowledge_eval import _explicit_abstention


def _read_trial(item: dict[str, Any]) -> RagTrial:
    raw_retrieval = item.get("retrieval")
    raw_agent_retrieval = item.get("agent_retrieval")

    def parse_result(raw: Any) -> RetrievalResult | None:
        if raw is None:
            return None
        chunks = tuple(
            RetrievedChunk(
                source_id=chunk["source_id"],
                text=chunk["text"],
                rank=chunk["rank"],
                reference=KnowledgeReference(**chunk["reference"]),
            )
            for chunk in raw["chunks"]
        )
        return RetrievalResult(
            profile=raw["profile"],
            query=raw["query"],
            chunks=chunks,
            latency_ms=raw["latency_ms"],
            request_id=raw["request_id"],
            error_code=raw.get("error_code"),
            error_message=raw.get("error_message"),
        )

    return RagTrial(
        case_id=item["case_id"],
        query=item["query"],
        profile=item["profile"],
        answerable=item["answerable"],
        expected_source_ids=tuple(item["expected_source_ids"]),
        expected_abstention_reason=item.get("expected_abstention_reason"),
        retrieved_source_ids=tuple(item["retrieved_source_ids"]),
        retrieval=parse_result(raw_retrieval),
        answer=item["answer"],
        knowledge_tool_called=item["knowledge_tool_called"],
        abstained=item["abstained"],
        agent_usage=item.get("agent_usage"),
        elapsed_ms=item["elapsed_ms"],
        error=item.get("error"),
        agent_retrieval=parse_result(raw_agent_retrieval),
        live_context=item.get("live_context"),
    )


def regrade(run_dir: Path, case_path: Path) -> dict[str, Any]:
    cases, case_hash = load_cases(case_path)
    repo_root = Path(__file__).resolve().parents[1]
    source_catalog = repo_root / "config/knowledge/source_catalog.yaml"
    knowledge_config = repo_root / "modules/config/knowledge.yaml"
    path = run_dir / "trial.jsonl"
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != len({item["case_id"] for item in records}):
        raise ValueError("trial records contain duplicate case IDs")
    selected = tuple(
        case for case in cases if case.case_id in {r["case_id"] for r in records}
    )
    if len(selected) != len(records):
        raise ValueError("trial file contains cases outside frozen set")
    for item in records:
        item["abstained_rule_v2"] = _explicit_abstention(item.get("answer", ""))
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    abstained = {item["case_id"]: item["abstained_rule_v2"] for item in records}
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["abstention"] = {
        key: {
            "numerator": metric.numerator,
            "denominator": metric.denominator,
            "value": metric.value,
        }
        for key, metric in compute_abstention_metrics(selected, abstained).items()
    }
    summary["abstention_method"] = "显式拒答/缺证措辞匹配 v2（非语义金标准）"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trials = tuple(
        replace(_read_trial(item), abstained=item["abstained_rule_v2"])
        for item in records
    )
    metadata_path = run_dir / "runtime_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("case_sha256", case_hash)
    metadata.setdefault(
        "source_sha256", hashlib.sha256(source_catalog.read_bytes()).hexdigest()
    )
    metadata["abstention_method"] = summary["abstention_method"]
    shutil.copy2(case_path, run_dir / "eval_cases.yaml")
    shutil.copy2(source_catalog, run_dir / "source_catalog.yaml")
    shutil.copy2(knowledge_config, run_dir / "knowledge.yaml")
    index_manifests: dict[str, Any] = {}
    for profile in ("vehicle_common", "vehiclemind_demo"):
        manifest = repo_root / "runs/lightrag" / profile / "storage/index_manifest.json"
        shutil.copy2(manifest, run_dir / f"{profile}_index_manifest.json")
        index_manifests[profile] = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["lightrag_version"] = "1.5.7"
    metadata["embedding_model"] = "text-embedding-v3"
    metadata["index_manifests"] = {
        profile: item["manifest_sha256"] for profile, item in index_manifests.items()
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.html").write_text(
        render_report(trials, summary, metadata), encoding="utf-8"
    )
    return summary["abstention"]


def main() -> int:
    parser = argparse.ArgumentParser(description="按更新后的明确弃答规则复算既有试次")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--cases", type=Path, default=Path("config/knowledge/eval_cases.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(regrade(args.run_dir, args.cases), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Opt-in online knowledge evaluation; ordinary replay never starts sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from modules.vehicle_ai.evaluation.rag_cases import RagCase, load_cases
from modules.vehicle_ai.evaluation.rag_judge import (
    review_abstention_trials,
    review_trials,
)
from modules.vehicle_ai.evaluation.rag_metrics import (
    compute_citation_support,
    compute_reviewed_abstention_metrics,
)
from modules.vehicle_ai.evaluation.rag_report import render_report
from modules.vehicle_ai.evaluation.rag_runner import AgentOutcome, run_cases
from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog
from modules.vehicle_ai.knowledge.lightrag_client import LightRAGClient
from modules.vehicle_ai.knowledge.models import RetrievalResult
from modules.vehicle_ai.knowledge.profile_router import ProfileRouter
from modules.vehicle_ai.llm import BaseLLMClient, LLMResponse, build_llm_client
from modules.vehicle_ai.runtime import VehicleMindRuntime
from scripts.build_knowledge_indexes import build_profile_index
from scripts.lightrag_services import (
    REPOSITORY_ROOT,
    build_service_specs,
    healthcheck_services,
    start_services,
    stop_services,
)


class _UsageClient(BaseLLMClient):
    def __init__(self, delegate: BaseLLMClient) -> None:
        self.delegate = delegate
        self.usage: dict[str, int] = {}

    def _capture(self, response: LLMResponse) -> LLMResponse:
        if response.usage:
            for key, value in response.usage.items():
                self.usage[key] = self.usage.get(key, 0) + value
        return response

    def chat(self, messages, tools=None) -> LLMResponse:
        return self._capture(self.delegate.chat(messages, tools))

    def chat_with_timeout(
        self, messages, tools=None, *, timeout_seconds: float
    ) -> LLMResponse:
        return self._capture(
            self.delegate.chat_with_timeout(
                messages, tools, timeout_seconds=timeout_seconds
            )
        )


class _CachedRetriever:
    def __init__(self, client: LightRAGClient) -> None:
        self.client = client
        self.cache: dict[tuple[str | None, str], RetrievalResult] = {}

    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        key = (profile, query)
        if key not in self.cache:
            self.cache[key] = self.client.query(profile, query, top_k=top_k)
        return self.cache[key]


class _AgentRetriever:
    def __init__(self, shared: _CachedRetriever) -> None:
        self.shared = shared
        self.last_result: RetrievalResult | None = None

    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        self.last_result = self.shared.query(profile, query, top_k=top_k)
        return self.last_result


def _explicit_abstention(answer: str) -> bool:
    """Conservative observable wording check, not an AI semantic judgment."""
    return any(
        phrase in answer
        for phrase in (
            "无法",
            "无法查询",
            "不能查询",
            "不能确认",
            "不能确定",
            "没有足够",
            "证据不足",
            "未包含",
            "不包含",
            "未提供",
            "未找到",
            "没有关于",
            "未涵盖",
            "信息中没有",
            "没有依据",
            "不知道",
            "资料不含",
        )
    )


def _wait_ready(specs) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if all(healthcheck_services(specs).values()):
            return
        time.sleep(1)
    raise TimeoutError("LightRAG 服务未在 90 秒内就绪")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VehicleMind 知识增强 Agent 在线评测")
    parser.add_argument("--mode", choices=("online-rag",), required=True)
    parser.add_argument("--provider", choices=("qwen", "glm"), required=True)
    parser.add_argument(
        "--profile",
        choices=("all", "vehicle_common", "vehiclemind_demo"),
        default="all",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--build-indexes", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--output-root", type=Path, default=Path("runs/rag_eval"))
    args = parser.parse_args(argv)

    root = REPOSITORY_ROOT
    load_dotenv(root / ".env")
    case_path = root / "config/knowledge/eval_cases.yaml"
    source_path = root / "config/knowledge/source_catalog.yaml"
    cases, case_hash = load_cases(case_path)
    sources = KnowledgeCatalog().load(source_path)
    source_profiles = {source.source_id: source.profile for source in sources}
    if args.profile != "all":
        cases = tuple(case for case in cases if case.profile == args.profile)
    if args.case_id:
        unknown = set(args.case_id) - {case.case_id for case in cases}
        if unknown:
            parser.error(
                f"未知或不在当前 profile 的 case: {', '.join(sorted(unknown))}"
            )
        cases = tuple(case for case in cases if case.case_id in args.case_id)
    if not cases:
        parser.error("没有选中的评测问题")
    if args.build_indexes:
        for profile in ("vehicle_common", "vehiclemind_demo"):
            build_profile_index(profile)
    specs = build_service_specs(root)
    status = healthcheck_services(specs)
    processes = {}
    if not all(status.values()):
        if any(status.values()):
            raise RuntimeError("两个 LightRAG 服务需同时就绪或同时停止")
        for profile, spec in specs.items():
            if not (spec.work_dir / "index_manifest.json").is_file():
                raise FileNotFoundError(
                    f"{profile} 尚未构建完整索引；使用 --build-indexes"
                )
        processes = start_services(specs, root=root)
    try:
        _wait_ready(specs)
        router = ProfileRouter()
        retriever = _CachedRetriever(LightRAGClient(router, sources))
        default_model = (
            os.getenv("QWEN_MODEL", "qwen3.8-max")
            if args.provider == "qwen"
            else os.getenv("GLM_MODEL", "glm-5.1")
        )
        model = args.model or default_model

        def answer(case: RagCase, result: RetrievalResult) -> AgentOutcome:
            llm = _UsageClient(
                build_llm_client(args.provider, model=model, timeout_seconds=120)
            )
            agent_retriever = _AgentRetriever(retriever)
            runtime = VehicleMindRuntime(
                llm,
                knowledge_profile=case.profile,
                knowledge_client=agent_retriever,
                turn_timeout_seconds=240,
                enable_event_recommendations=False,
            )
            final = runtime.agent.chat(case.query)
            called = any(
                record.name == "search_vehicle_knowledge"
                for record in runtime.tools.execution_history()
            )
            live_context: object = next(
                (
                    record.result_data.get("live_context", {})
                    for record in reversed(runtime.tools.execution_history())
                    if record.name == "search_vehicle_knowledge"
                ),
                {},
            )
            return AgentOutcome(
                final,
                called,
                _explicit_abstention(final),
                llm.usage or None,
                agent_retriever.last_result,
                live_context if isinstance(live_context, dict) else {},
            )

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = args.output_root / f"{stamp}-{args.provider}-{args.profile}"
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(case_path, destination / "eval_cases.yaml")
        shutil.copy2(source_path, destination / "source_catalog.yaml")
        for profile, spec in specs.items():
            shutil.copy2(
                spec.work_dir / "index_manifest.json",
                destination / f"{profile}_index_manifest.json",
            )
        index_manifest_data = {
            profile: json.loads(
                (destination / f"{profile}_index_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for profile in specs
        }
        trials, summary = run_cases(
            cases, retriever, answer, destination, source_profiles=source_profiles
        )
        judge = build_llm_client(
            args.provider, model=model, timeout_seconds=120, temperature=0
        )
        reviews, review_errors = review_trials(judge, trials)
        with (destination / "citation_reviews.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for review in reviews:
                output.write(json.dumps(asdict(review), ensure_ascii=False) + "\n")
        support = compute_citation_support(reviews)
        summary["citation_support"] = {
            "numerator": support.numerator,
            "denominator": support.denominator,
            "value": support.value,
        }
        summary["citation_review_status"] = (
            "partial_ai_assisted" if review_errors else "ai_assisted_internal"
        )
        summary["citation_review_errors"] = review_errors
        explicit_metrics = summary["abstention"]
        abstention_reviews, abstention_errors = review_abstention_trials(judge, trials)
        with (destination / "abstention_reviews.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for review_item in abstention_reviews:
                output.write(json.dumps(asdict(review_item), ensure_ascii=False) + "\n")
        summary["explicit_wording_match"] = explicit_metrics
        semantic_metrics = compute_reviewed_abstention_metrics(
            cases, abstention_reviews
        )
        summary["abstention"] = {
            key: {
                "numerator": metric.numerator,
                "denominator": metric.denominator,
                "value": metric.value,
            }
            for key, metric in semantic_metrics.items()
        }
        summary["abstention_review_status"] = (
            "partial_ai_assisted" if abstention_errors else "ai_assisted_semantic"
        )
        summary["abstention_review_errors"] = abstention_errors
        (destination / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata = {
            "provider": args.provider,
            "model": model,
            "profile": args.profile,
            "case_sha256": case_hash,
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "abstention_method": "显式证据不足措辞匹配（非语义金标准）",
            "case_scope": f"{len(cases)}/30",
            "abstention_reviewer": {"provider": args.provider, "model": model},
            "lightrag_version": index_manifest_data["vehicle_common"][
                "lightrag_version"
            ],
            "embedding_model": "text-embedding-v3",
            "index_manifests": {
                profile: data["manifest_sha256"]
                for profile, data in index_manifest_data.items()
            },
        }
        (destination / "runtime_manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (destination / "report.html").write_text(
            render_report(trials, summary, metadata), encoding="utf-8"
        )
        print(f"评测报告：{destination / 'report.html'}")
        print(f"试次数：{len(trials)}；失败：{summary['failed_trials']}")
        return 0
    finally:
        stop_services(processes)


if __name__ == "__main__":
    raise SystemExit(main())

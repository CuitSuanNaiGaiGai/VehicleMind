from pathlib import Path

from modules.vehicle_ai.evaluation.rag_cases import load_cases
from modules.vehicle_ai.evaluation.rag_report import render_report
from modules.vehicle_ai.evaluation.rag_runner import AgentOutcome, RagTrial, run_cases
from modules.vehicle_ai.knowledge.models import (
    KnowledgeReference,
    RetrievedChunk,
    RetrievalResult,
)


def test_report_prioritizes_answer_and_evidence_in_chinese() -> None:
    ref = KnowledgeReference(
        "K001", "困倦线索", "安全建议", "https://example.org/k1", "vehicle_common"
    )
    result = RetrievalResult(
        "vehicle_common",
        "困了怎么办",
        (RetrievedChunk("K001", "应安全停车休息", 1, ref),),
        20.0,
        "r1",
    )
    trial = RagTrial(
        "R01",
        "困了怎么办",
        "vehicle_common",
        True,
        ("K001",),
        None,
        ("K001",),
        result,
        "应停车休息。[K001]",
        True,
        False,
        {"total_tokens": 100},
        300.0,
        None,
        live_context={"driver.risk": "HIGH"},
    )
    report = render_report(
        (trial,),
        {
            "case_count": 1,
            "recall_at_5": {"numerator": 1, "denominator": 1, "value": 1.0},
            "citation_support": {"numerator": 0, "denominator": 0, "value": None},
            "profile_leakage": 0,
            "abstention": {},
        },
        {"case_sha256": "abc", "source_sha256": "def"},
    )
    for phrase in (
        "应停车休息。[K001]",
        "困倦线索",
        "应安全停车休息",
        "证据来源",
        "工具调用",
        "实时状态来源",
        "driver.risk=HIGH",
        "20.0",
        "100",
        "Recall@5（前五条证据召回率）",
        "Citation Support（引用支持率）",
        "Profile Leakage（检索来源范围泄漏）",
        "AI 辅助内部评测",
        "https://example.org/k1",
    ):
        assert phrase in report


def test_frozen_30_case_fixture_emits_report_and_raw_artifacts(tmp_path: Path) -> None:
    class EmptyRetriever:
        def query(
            self, profile: str | None, query: str, *, top_k: int = 5
        ) -> RetrievalResult:
            return RetrievalResult(
                profile or "vehicle_common", query, (), 1.0, "fixture"
            )

    path = Path(__file__).resolve().parents[3] / "config/knowledge/eval_cases.yaml"
    cases, digest = load_cases(path)
    trials, summary = run_cases(
        cases,
        EmptyRetriever(),
        lambda case, result: AgentOutcome("证据不足。", True, True, None),
        tmp_path,
    )
    (tmp_path / "report.html").write_text(
        render_report(trials, summary, {"case_sha256": digest}), encoding="utf-8"
    )
    assert summary["case_count"] == 30
    assert sum(1 for _ in (tmp_path / "trial.jsonl").open()) == 30
    assert (tmp_path / "summary.json").is_file()
    assert "R30" in (tmp_path / "report.html").read_text(encoding="utf-8")

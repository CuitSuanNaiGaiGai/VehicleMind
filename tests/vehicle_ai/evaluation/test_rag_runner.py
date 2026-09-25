import json
from pathlib import Path

from modules.vehicle_ai.evaluation.rag_cases import RagCase
from modules.vehicle_ai.evaluation.rag_runner import AgentOutcome, run_cases
from modules.vehicle_ai.knowledge.models import RetrievalResult


class FakeRetriever:
    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        return RetrievalResult(profile or "vehicle_common", query, (), 5.0, "request-1")


def test_runner_persists_failed_trials_and_unavailable_usage(tmp_path: Path) -> None:
    cases = (
        RagCase("A", "可回答", "vehicle_common", True, ("K001",)),
        RagCase("B", "无法回答", "vehicle_common", False, (), "no_answer"),
    )

    def answer(case: RagCase, result: RetrievalResult) -> AgentOutcome:
        if case.case_id == "A":
            raise RuntimeError("model unavailable")
        return AgentOutcome("当前没有依据。", False, True, None)

    trials, summary = run_cases(cases, FakeRetriever(), answer, tmp_path)
    assert len(trials) == 2 and trials[0].error == "model unavailable"
    assert trials[0].agent_usage is None
    assert summary["recall_at_5"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    records = [
        json.loads(line) for line in (tmp_path / "trial.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2 and records[0]["error"] == "model unavailable"
    assert records[0]["agent_usage"] is None

from __future__ import annotations

from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.knowledge.models import (
    KnowledgeReference,
    RetrievedChunk,
    RetrievalResult,
)
from modules.vehicle_ai.knowledge.tool import build_knowledge_tool


class FakeClient:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple[str | None, str, int]] = []

    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        self.calls.append((profile, query, top_k))
        return self.result


def _result(*, error: str | None = None) -> RetrievalResult:
    ref = KnowledgeReference(
        "K001", "疲劳驾驶", "安全建议", "https://example.org", "vehicle_common"
    )
    return RetrievalResult(
        profile="vehicle_common",
        query="疲劳怎么办",
        chunks=() if error else (RetrievedChunk("K001", "需要安全停车休息", 1, ref),),
        latency_ms=12.0,
        request_id="request-1",
        error_code=error,
    )


def test_tool_is_read_only_and_keeps_profile_out_of_schema() -> None:
    client = FakeClient(_result())
    tool = build_knowledge_tool(client, ContextManager(), "vehicle_common")
    assert tool.read_only and not tool.requires_confirmation
    assert set(tool.parameters["properties"]) == {"query"}
    assert tool.parameters["additionalProperties"] is False
    result = tool.handler(query="疲劳怎么办")
    assert result.success
    assert result.data["evidence"][0]["source_id"] == "K001"
    assert "[K001]" in result.message
    assert client.calls[0][0] == "vehicle_common"


def test_only_known_relevant_context_enters_query() -> None:
    now = [0.0]
    manager = ContextManager(quality_clock=lambda: now[0])
    manager.update_driver(eye_closure_seconds=2.7, recent_yawns=2)
    manager.update_vehicle(speed_kmh=42.0)
    client = FakeClient(_result())
    tool = build_knowledge_tool(client, manager, "vehicle_common")
    result = tool.handler(query="司机闭眼该如何休息")
    assert result.data["live_context"]["driver.eye_closure_seconds"] == 2.7
    assert "vehicle.speed_kmh" not in result.data["live_context"]
    assert "driver.eye_closure_seconds=2.7" in client.calls[0][1]
    now[0] = 3.0
    tool.handler(query="司机闭眼该如何休息")
    assert "driver.eye_closure_seconds" not in client.calls[1][1]


def test_retrieval_failure_is_explicit_without_evidence() -> None:
    result = build_knowledge_tool(
        FakeClient(_result(error="timeout")), ContextManager(), None
    ).handler(query="疲劳怎么办")
    assert not result.success
    assert result.error == "KNOWLEDGE_UNAVAILABLE"
    assert result.data["evidence"] == []

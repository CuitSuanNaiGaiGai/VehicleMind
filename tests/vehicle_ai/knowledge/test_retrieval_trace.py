"""Keep evidence and errors in an append-only local trace."""

from __future__ import annotations

import json
from pathlib import Path

from modules.vehicle_ai.knowledge.models import RetrievalResult
from modules.vehicle_ai.knowledge.trace import record_retrieval_trace


def test_trace_preserves_failed_and_successful_queries(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    failed = RetrievalResult(
        profile="vehicle_common",
        query="为什么要休息？",
        chunks=(),
        latency_ms=12.5,
        request_id="req-1",
        error_code="timeout",
        error_message="service timed out",
    )
    succeeded = RetrievalResult(
        profile="vehiclemind_demo",
        query="如何播放音乐？",
        chunks=(),
        latency_ms=8.0,
        request_id="req-2",
    )

    record_retrieval_trace(path, failed)
    record_retrieval_trace(path, succeeded)

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["error_code"] == "timeout"
    assert lines[1]["profile"] == "vehiclemind_demo"
    assert lines[0]["latency_ms"] == 12.5

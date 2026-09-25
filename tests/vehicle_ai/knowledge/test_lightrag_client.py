"""Only catalog-mapped LightRAG evidence can reach the Agent."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

from modules.vehicle_ai.knowledge.catalog import KnowledgeCatalog
from modules.vehicle_ai.knowledge.lightrag_client import LightRAGClient
from modules.vehicle_ai.knowledge.profile_router import ProfileRouter

ROOT = Path(__file__).resolve().parents[3]
CATALOG = KnowledgeCatalog().load(ROOT / "config/knowledge/source_catalog.yaml")


class FakeResponse(io.BytesIO):
    status = 200


class FakeOpener:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.requests: list[Request] = []

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        assert timeout > 0
        self.requests.append(request)
        if isinstance(self.payload, Exception):
            raise self.payload
        return FakeResponse(json.dumps(self.payload).encode())


def _response(filename: str = "K001-yawn-blink.md") -> dict:
    return {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "content": "频繁哈欠是困倦线索",
                    "file_path": filename,
                    "reference_id": "1",
                    "chunk_id": "chunk-1",
                }
            ],
            "references": [{"reference_id": "1", "file_path": filename}],
        },
        "metadata": {"query_mode": "mix"},
        "response": "LightRAG 自行生成的文本不得转交给 Agent",
    }


def _client(opener: FakeOpener) -> LightRAGClient:
    return LightRAGClient(ProfileRouter(), CATALOG, opener=opener)


def test_query_sends_context_only_request_and_maps_citation() -> None:
    opener = FakeOpener(_response())

    result = _client(opener).query("vehicle_common", "疲劳时为什么要休息？")

    assert result.error_code is None
    assert result.profile == "vehicle_common"
    assert result.chunks[0].source_id == "K001"
    assert result.chunks[0].reference.title == "疲劳驾驶的眼部与哈欠线索"
    assert "LightRAG 自行生成" not in result.chunks[0].text
    request = opener.requests[0]
    assert request.full_url == "http://127.0.0.1:9621/query/data"
    body = json.loads(request.data)
    assert body == {
        "query": "疲劳时为什么要休息？",
        "mode": "mix",
        "only_need_context": True,
        "include_references": True,
        "include_chunk_content": True,
        "enable_rerank": False,
        "top_k": 5,
        "chunk_top_k": 5,
    }


def test_unknown_profile_cannot_query_demo_endpoint_or_cite_demo_source() -> None:
    opener = FakeOpener(_response("K011-cabin-evidence.md"))

    result = _client(opener).query("http://evil.example", "车机是什么？")

    assert opener.requests[0].full_url == "http://127.0.0.1:9621/query/data"
    assert result.error_code == "unknown_reference"
    assert result.chunks == ()


def test_demo_profile_may_use_common_evidence() -> None:
    result = _client(FakeOpener(_response())).query(
        "vehiclemind_demo", "打哈欠意味着什么？"
    )

    assert result.profile == "vehiclemind_demo"
    assert result.chunks[0].source_id == "K001"


def test_reference_id_mismatch_fails_closed() -> None:
    payload = _response()
    payload["data"]["chunks"][0]["reference_id"] = "other"

    result = _client(FakeOpener(payload)).query("vehicle_common", "什么是困倦？")

    assert result.error_code == "unknown_reference"
    assert result.chunks == ()


def test_malformed_response_is_observable_error() -> None:
    result = _client(FakeOpener({"status": "success", "data": []})).query(
        "vehicle_common", "什么是困倦？"
    )

    assert result.error_code == "invalid_response"
    assert result.chunks == ()


def test_timeout_is_observable_error() -> None:
    result = _client(FakeOpener(TimeoutError("slow"))).query(
        "vehicle_common", "什么是困倦？"
    )

    assert result.error_code == "timeout"
    assert result.chunks == ()


def test_unavailable_service_is_observable_error() -> None:
    result = _client(FakeOpener(URLError("offline"))).query(
        "vehicle_common", "什么是困倦？"
    )

    assert result.error_code == "service_unavailable"
    assert result.chunks == ()


def test_query_limits_output_to_five_chunks() -> None:
    payload = _response()
    payload["data"]["chunks"] *= 9
    opener = FakeOpener(payload)

    result = _client(opener).query("vehicle_common", "什么是困倦？", top_k=50)

    assert len(result.chunks) == 5
    assert json.loads(opener.requests[0].data)["chunk_top_k"] == 5

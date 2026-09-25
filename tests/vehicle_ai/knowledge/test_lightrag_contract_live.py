"""Opt-in contract smoke test against the pinned local LightRAG server."""

from __future__ import annotations

import json
import os
import time
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest


def _json_request(
    url: str, payload: dict[str, object] | None = None, *, timeout: float = 45
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urlopen(request, timeout=timeout) as response:
        assert response.status == 200
        return json.load(response)


@pytest.mark.online
@pytest.mark.parametrize("port", [9621, 9622])
def test_upload_status_and_context_only_query_contract(port: int) -> None:
    if os.getenv("VEHICLEMIND_LIGHTRAG_LIVE") != "1":
        pytest.skip("set VEHICLEMIND_LIGHTRAG_LIVE=1 to test local services")
    base = f"http://127.0.0.1:{port}"
    assert _json_request(f"{base}/health")

    nonce = uuid4().hex[:12]
    filename = f"K999-contract-{nonce}.md"
    content = f"VehicleMind 测试知识：青梅路测试停车点编号为 {nonce}。".encode()
    boundary = f"vehiclemind-{nonce}"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/markdown\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = Request(
        f"{base}/documents/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urlopen(request, timeout=45) as response:
        upload = json.load(response)
    assert upload["status"] == "success"
    track_id = upload["track_id"]

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        track = _json_request(f"{base}/documents/track_status/{track_id}")
        statuses = {item["status"] for item in track["documents"]}
        if "failed" in statuses:
            pytest.fail(f"LightRAG indexing failed: {track['status_summary']}")
        if statuses == {"processed"}:
            break
        time.sleep(3)
    else:
        pytest.fail("LightRAG did not finish indexing within 180 seconds")

    result = _json_request(
        f"{base}/query/data",
        {
            "query": f"青梅路测试停车点的编号 {nonce} 是什么？",
            "mode": "mix",
            "only_need_context": True,
            "include_references": True,
            "include_chunk_content": True,
            "enable_rerank": False,
        },
        timeout=180,
    )
    assert result["status"] == "success"
    assert result["metadata"]["query_mode"] == "mix"
    assert isinstance(result["data"]["chunks"], list)
    assert isinstance(result["data"]["references"], list)
    assert any(filename in item["file_path"] for item in result["data"]["references"])

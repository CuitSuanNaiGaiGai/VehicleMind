"""Retrieve scoped source chunks from LightRAG's structured data API."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from modules.vehicle_ai.knowledge.models import (
    KnowledgeReference,
    KnowledgeSource,
    RetrievedChunk,
    RetrievalResult,
)
from modules.vehicle_ai.knowledge.profile_router import ProfileRouter

_SOURCE_FILENAME = re.compile(r"^(K[0-9]{3,})-.+\.md$")


class _RetrievalFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class LightRAGClient:
    def __init__(
        self,
        router: ProfileRouter,
        sources: Sequence[KnowledgeSource],
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.router = router
        self.sources = {source.source_id: source for source in sources}
        self.opener = opener

    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        effective_profile, base_url = self.router.resolve(profile)
        request_id = uuid4().hex
        started = time.perf_counter()
        limit = max(1, min(top_k, self.router.top_k))
        try:
            if not query.strip():
                raise _RetrievalFailure("invalid_query", "knowledge query is empty")
            body = {
                "query": query.strip(),
                "mode": "mix",
                "only_need_context": True,
                "include_references": True,
                "include_chunk_content": True,
                "enable_rerank": False,
                "top_k": limit,
                "chunk_top_k": limit,
            }
            request = Request(
                f"{base_url}/query/data",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with self.opener(request, timeout=self.router.timeout_seconds) as response:
                payload = json.load(response)
            chunks = self._parse(payload, effective_profile, limit)
            error_code = None
            error_message = None
        except _RetrievalFailure as error:
            chunks = ()
            error_code = error.code
            error_message = str(error)
        except (TimeoutError, ConnectionError) as error:
            chunks = ()
            error_code = "timeout"
            error_message = str(error)
        except HTTPError as error:
            chunks = ()
            error_code = (
                "index_unavailable" if error.code >= 500 else "invalid_response"
            )
            error_message = f"LightRAG HTTP {error.code}"
        except URLError as error:
            chunks = ()
            error_code = "service_unavailable"
            error_message = str(error.reason)
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as error:
            chunks = ()
            error_code = "invalid_response"
            error_message = str(error)
        return RetrievalResult(
            profile=effective_profile,
            query=query,
            chunks=chunks,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            request_id=request_id,
            error_code=error_code,
            error_message=error_message,
        )

    def _parse(
        self, payload: Any, profile: str, limit: int
    ) -> tuple[RetrievedChunk, ...]:
        if not isinstance(payload, dict):
            raise _RetrievalFailure("invalid_response", "response is not an object")
        if payload.get("status") != "success":
            raise _RetrievalFailure("index_unavailable", "LightRAG query failed")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise _RetrievalFailure("invalid_response", "data is not an object")
        raw_chunks = data.get("chunks")
        raw_references = data.get("references")
        if not isinstance(raw_chunks, list) or not isinstance(raw_references, list):
            raise _RetrievalFailure("invalid_response", "chunks/references are missing")
        references: dict[str, str] = {}
        for item in raw_references:
            if not isinstance(item, dict):
                raise _RetrievalFailure(
                    "invalid_response", "reference is not an object"
                )
            reference_id = item.get("reference_id")
            file_path = item.get("file_path")
            if not isinstance(reference_id, str) or not isinstance(file_path, str):
                raise _RetrievalFailure(
                    "invalid_response", "reference fields are invalid"
                )
            if reference_id in references and references[reference_id] != file_path:
                raise _RetrievalFailure("invalid_response", "ambiguous reference ID")
            references[reference_id] = file_path
        chunks: list[RetrievedChunk] = []
        for rank, item in enumerate(raw_chunks[:limit], 1):
            if not isinstance(item, dict):
                raise _RetrievalFailure("invalid_response", "chunk is not an object")
            text = item.get("content")
            file_path = item.get("file_path")
            reference_id = item.get("reference_id")
            if (
                not isinstance(text, str)
                or not isinstance(file_path, str)
                or not isinstance(reference_id, str)
            ):
                raise _RetrievalFailure("invalid_response", "chunk fields are invalid")
            if (
                reference_id not in references
                or Path(references[reference_id]).name != Path(file_path).name
            ):
                raise _RetrievalFailure(
                    "unknown_reference", "chunk citation is unmapped"
                )
            filename = Path(file_path).name
            match = _SOURCE_FILENAME.fullmatch(filename)
            source = self.sources.get(match.group(1)) if match else None
            if (
                source is None
                or Path(source.text_path).name != filename
                or (profile == "vehicle_common" and source.profile != "vehicle_common")
            ):
                raise _RetrievalFailure(
                    "unknown_reference", "source is outside catalog scope"
                )
            reference = KnowledgeReference(
                source_id=source.source_id,
                title=source.title,
                section=source.section,
                source_uri=source.source_uri,
                profile=source.profile,
            )
            chunks.append(
                RetrievedChunk(
                    source_id=source.source_id,
                    text=text,
                    rank=rank,
                    reference=reference,
                )
            )
        return tuple(chunks)

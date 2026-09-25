"""Read-only knowledge tool; the Agent, not LightRAG, writes the final answer."""

from __future__ import annotations

from typing import Any, Protocol

from modules.vehicle_ai.context import ContextManager
from modules.vehicle_ai.knowledge.models import RetrievalResult
from modules.vehicle_ai.tools.base import ToolDefinition, ToolResult

_CONTEXT_GROUPS: tuple[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    (
        ("疲劳", "困", "闭眼", "哈欠", "休息", "睡"),
        (
            ("driver", "state"),
            ("driver", "risk"),
            ("driver", "eye_closure_seconds"),
            ("driver", "recent_yawns"),
        ),
    ),
    (
        ("道路", "车道", "路况", "交通", "行人"),
        (
            ("road", "traffic_level"),
            ("road", "lane_detected"),
            ("road", "pedestrian_count"),
        ),
    ),
    (("速度", "行驶", "停车", "车速"), (("vehicle", "speed_kmh"),)),
)


class KnowledgeClient(Protocol):
    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult: ...


def _known_context(manager: ContextManager, query: str) -> dict[str, Any]:
    fields = tuple(
        field
        for keywords, group in _CONTEXT_GROUPS
        if any(keyword in query for keyword in keywords)
        for field in group
    )
    if not fields:
        return {}
    snapshot, qualities = manager.snapshot_with_quality(fields)
    return {
        f"{domain}.{field}": getattr(getattr(snapshot, domain), field)
        for domain, field in fields
        if qualities[(domain, field)] == "KNOWN"
    }


def build_knowledge_tool(
    client: KnowledgeClient,
    context_manager: ContextManager,
    profile: str | None,
) -> ToolDefinition:
    """Build one profile-bound tool with a query-only model-facing schema."""

    def search_vehicle_knowledge(query: str) -> ToolResult:
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 1000:
            return ToolResult(False, "知识检索问题无效。", error="INVALID_QUERY")
        live = _known_context(context_manager, query)
        suffix = (
            "\n当前已知车辆状态："
            + "；".join(f"{key}={value}" for key, value in live.items())
            if live
            else ""
        )
        retrieved = client.query(profile, query.strip() + suffix, top_k=5)
        data: dict[str, Any] = {
            "profile": retrieved.profile,
            "request_id": retrieved.request_id,
            "latency_ms": retrieved.latency_ms,
            "live_context": live,
            "evidence": [
                {
                    "source_id": chunk.source_id,
                    "title": chunk.reference.title,
                    "section": chunk.reference.section,
                    "source_uri": chunk.reference.source_uri,
                    "text": chunk.text,
                    "rank": chunk.rank,
                }
                for chunk in retrieved.chunks[:5]
            ],
        }
        if retrieved.error_code:
            data["retrieval_error"] = retrieved.error_code
            return ToolResult(
                False,
                "知识证据暂不可用，不要据此编造答案。",
                data,
                error="KNOWLEDGE_UNAVAILABLE",
            )
        if not retrieved.chunks:
            return ToolResult(True, "未检索到匹配的知识证据。", data)
        citations = "、".join(f"[{item['source_id']}]" for item in data["evidence"])
        return ToolResult(True, f"检索到知识证据 {citations}。", data)

    return ToolDefinition(
        name="search_vehicle_knowledge",
        description="按需检索车辆知识，仅返回可引用证据，不执行车机动作。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=search_vehicle_knowledge,
        category="knowledge",
        read_only=True,
    )

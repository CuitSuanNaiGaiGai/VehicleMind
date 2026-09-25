from __future__ import annotations

import json

from modules.vehicle_ai.knowledge.models import RetrievalResult
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime


class FakeKnowledgeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str]] = []

    def query(
        self, profile: str | None, query: str, *, top_k: int = 5
    ) -> RetrievalResult:
        self.calls.append((profile, query))
        return RetrievalResult(
            profile="vehicle_common",
            query=query,
            chunks=(),
            latency_ms=1,
            request_id="r1",
        )


def test_default_runtime_does_not_expose_knowledge_or_call_service() -> None:
    llm = ScriptedLLMClient((ScriptedResponse(content="你好"),))
    runtime = VehicleMindRuntime(llm=llm)
    assert "search_vehicle_knowledge" not in runtime.tools.names()
    assert runtime.agent.chat("你好") == "你好"


def test_opt_in_agent_calls_read_only_knowledge_without_pending_action() -> None:
    arguments = {"query": "疲劳驾驶应该如何休息"}
    llm = ScriptedLLMClient(
        (
            ScriptedResponse(
                content=None,
                tool_calls=(
                    LLMToolCall(
                        "c1",
                        "search_vehicle_knowledge",
                        arguments,
                        json.dumps(arguments),
                    ),
                ),
            ),
            ScriptedResponse(content="目前没有可引用的知识证据。"),
        )
    )
    client = FakeKnowledgeClient()
    runtime = VehicleMindRuntime(
        llm=llm, knowledge_profile="vehicle_common", knowledge_client=client
    )
    assert "search_vehicle_knowledge" in runtime.tools.names()
    assert runtime.tools.get("search_vehicle_knowledge").read_only
    assert runtime.agent.chat("疲劳驾驶应该如何休息") == "目前没有可引用的知识证据。"
    assert len(client.calls) == 1 and client.calls[0][0] == "vehicle_common"
    assert runtime.agent.pending_actions.get() is None
    assert llm.requests[0].tools is not None
    assert any(
        tool["function"]["name"] == "search_vehicle_knowledge"
        for tool in llm.requests[0].tools
    )


def test_model_cannot_select_profile_or_endpoint() -> None:
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient((ScriptedResponse(content="unused"),)),
        knowledge_profile="vehicle_common",
        knowledge_client=FakeKnowledgeClient(),
    )
    result = runtime.tools.execute(
        "search_vehicle_knowledge", {"query": "疲劳", "profile": "vehiclemind_demo"}
    )
    assert not result.success and result.error == "UNKNOWN_ARGUMENTS"

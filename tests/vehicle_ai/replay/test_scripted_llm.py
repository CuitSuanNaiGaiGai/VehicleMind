from __future__ import annotations

import pytest

from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient


def _responses() -> tuple[ScriptedResponse, ...]:
    return (
        ScriptedResponse(
            content=None,
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="search_nearby_rest_area",
                    arguments={"radius_km": 10},
                    arguments_json='{"radius_km": 10}',
                ),
            ),
        ),
        ScriptedResponse(content="Please confirm navigation."),
    )


def test_scripted_llm_consumes_responses_in_order() -> None:
    client = ScriptedLLMClient(responses=_responses())

    first = client.chat(messages=[{"role": "user", "content": "Find a rest area"}])
    second = client.chat(messages=[{"role": "tool", "content": "found"}])

    assert first.content is None
    assert first.tool_calls[0].id == "call-1"
    assert first.tool_calls[0].name == "search_nearby_rest_area"
    assert first.tool_calls[0].arguments == {"radius_km": 10}
    assert second.content == "Please confirm navigation."
    assert second.tool_calls == []
    assert client.remaining == 0


def test_scripted_llm_rejects_an_empty_response_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ScriptedLLMClient(responses=())


def test_scripted_llm_raises_when_exhausted() -> None:
    client = ScriptedLLMClient(responses=(ScriptedResponse(content="Only response"),))
    client.chat(messages=[])

    with pytest.raises(RuntimeError, match="exhausted"):
        client.chat(messages=[])


def test_requests_and_responses_are_defensive_copies() -> None:
    source = _responses()
    messages = [{"role": "user", "content": "Find a rest area"}]
    tools = [{"type": "function", "function": {"name": "search"}}]
    client = ScriptedLLMClient(responses=source)

    response = client.chat(messages=messages, tools=tools)
    messages[0]["content"] = "mutated"
    tools[0]["function"]["name"] = "mutated"
    response.tool_calls[0].arguments["radius_km"] = 999

    request = client.requests[0]
    assert request.messages[0]["content"] == "Find a rest area"
    assert request.tools is not None
    assert request.tools[0]["function"]["name"] == "search"
    assert source[0].tool_calls[0].arguments == {"radius_km": 10}


def test_requests_property_does_not_expose_internal_history() -> None:
    client = ScriptedLLMClient(responses=_responses())
    client.chat(messages=[{"role": "user", "content": "hello"}])

    exported = client.requests
    exported[0].messages[0]["content"] = "changed"

    assert client.requests[0].messages[0]["content"] == "hello"

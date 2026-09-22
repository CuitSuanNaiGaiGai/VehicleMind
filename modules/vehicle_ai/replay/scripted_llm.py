from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse, LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse


@dataclass
class ScriptedRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None


def _copy_response(response: ScriptedResponse) -> ScriptedResponse:
    calls = tuple(
        LLMToolCall(
            id=call.id,
            name=call.name,
            arguments=deepcopy(call.arguments),
            arguments_json=call.arguments_json,
        )
        for call in response.tool_calls
    )
    return ScriptedResponse(content=response.content, tool_calls=calls)


class ScriptedLLMClient(BaseLLMClient):
    """Deterministic provider-compatible client for offline replay."""

    def __init__(self, responses: tuple[ScriptedResponse, ...]):
        if not responses:
            raise ValueError("scripted LLM requires at least one response")
        self._responses = deque(_copy_response(item) for item in responses)
        self._requests: list[ScriptedRequest] = []

    @property
    def remaining(self) -> int:
        return len(self._responses)

    @property
    def requests(self) -> list[ScriptedRequest]:
        return deepcopy(self._requests)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self._requests.append(
            ScriptedRequest(
                messages=deepcopy(messages),
                tools=deepcopy(tools),
            )
        )
        if not self._responses:
            raise RuntimeError("scripted LLM response queue is exhausted")

        scripted = self._responses.popleft()
        tool_calls = [
            LLMToolCall(
                id=call.id,
                name=call.name,
                arguments=deepcopy(call.arguments),
                arguments_json=call.arguments_json,
            )
            for call in scripted.tool_calls
        ]
        return LLMResponse(
            content=scripted.content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

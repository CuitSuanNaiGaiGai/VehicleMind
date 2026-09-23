from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMToolCall:
    """
    Provider-independent tool call.
    """

    id: str

    name: str

    arguments: dict[str, Any]

    arguments_json: str


@dataclass
class LLMResponse:
    """
    Normalized LLM response.
    """

    content: str | None

    tool_calls: list[LLMToolCall]

    finish_reason: str | None = None

    response_model: str | None = None

    usage: dict[str, int] | None = None


class BaseLLMClient(ABC):
    """
    Common interface for Qwen / GLM.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:

        raise NotImplementedError

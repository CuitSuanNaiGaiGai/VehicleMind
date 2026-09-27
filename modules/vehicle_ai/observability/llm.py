from __future__ import annotations

import time
from typing import Any

from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse
from modules.vehicle_ai.observability.trace import AgentTraceRecorder


class TracingLLMClient(BaseLLMClient):
    """Measure model calls without recording prompts or response bodies."""

    def __init__(self, inner: BaseLLMClient, recorder: AgentTraceRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "chat":
            # Treat per-instance chat replacements as wrapper hooks. Forwarding
            # a wrapper-bound method to inner.chat would make it call itself.
            object.__setattr__(self, name, value)
            return
        inner = self.__dict__.get("inner")
        if (
            name not in {"inner", "recorder"}
            and inner is not None
            and hasattr(inner, name)
        ):
            setattr(inner, name, value)
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Preserve provider-specific diagnostics and test hooks exposed by clients.
        return getattr(self.inner, name)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        return self._call(lambda: self.inner.chat(messages, tools))

    def chat_with_timeout(
        self,
        messages,
        tools=None,
        *,
        timeout_seconds: float,
    ) -> LLMResponse:
        if "chat" in self.__dict__:
            # Wrapping hooks commonly call the saved TracingLLMClient.chat
            # method, which records the underlying call itself.
            return self.__dict__["chat"](messages, tools)
        return self._call(
            lambda: self.inner.chat_with_timeout(
                messages, tools, timeout_seconds=timeout_seconds
            )
        )

    def _call(self, invoke) -> LLMResponse:
        started = time.perf_counter()
        attributes: dict[str, Any] = {"status": "COMPLETED"}
        try:
            response = invoke()
            attributes.update(
                response_model=response.response_model,
                tool_call_count=len(response.tool_calls),
                tool_names=[call.name for call in response.tool_calls],
            )
            return response
        except Exception as error:
            attributes.update(
                status="FAILED",
                error_type=type(error).__name__,
            )
            raise
        finally:
            self.recorder.emit(
                "model_call",
                latency_ms=(time.perf_counter() - started) * 1000,
                attributes=attributes,
            )

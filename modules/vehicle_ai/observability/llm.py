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

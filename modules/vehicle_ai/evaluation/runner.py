from __future__ import annotations

import time
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from modules.vehicle_ai.context import GearState, NavigationState
from modules.vehicle_ai.evaluation.models import EvaluationCase
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse
from modules.vehicle_ai.replay.trace import plain_value
from modules.vehicle_ai.runtime import VehicleMindRuntime
from modules.observation import ObservationMetadata
from modules.vehicle_ai.tools.base import ToolResult


@dataclass(frozen=True)
class TrialResult:
    case_id: str
    provider: str
    model: str
    trial_index: int
    replies: tuple[str, ...]
    tool_calls: tuple[dict[str, Any], ...]
    final_context: dict[str, Any]
    request_count: int
    latency_ms: float
    error: str | None
    model_responses: tuple[dict[str, Any], ...]
    case_sha256: str
    tool_schema_sha256: str
    request_latencies_ms: tuple[float, ...]
    requested_tools: tuple[dict[str, Any], ...]
    requests: tuple[dict[str, Any], ...]
    settings: dict[str, Any]
    interaction_events: tuple[dict[str, Any], ...]


class RecordingClient(BaseLLMClient):
    def __init__(self, inner: BaseLLMClient):
        self.inner = inner
        self.requests: list[dict[str, Any]] = []
        self.responses: list[LLMResponse] = []
        self.latencies_ms: list[float] = []

    def chat(self, messages, tools=None) -> LLMResponse:
        self.requests.append(
            {
                "at_ms": int(time.time() * 1000),
                "messages": plain_value(messages),
                "tools": plain_value(tools or []),
            }
        )
        started = time.perf_counter()
        try:
            response = self.inner.chat(messages, tools)
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1000)
        self.responses.append(response)
        return response


def run_trial(
    case: EvaluationCase,
    client: BaseLLMClient,
    *,
    provider: str,
    model: str,
    trial_index: int,
    max_tool_rounds: int = 5,
) -> TrialResult:
    """Run one isolated trial; only the supplied client may access a provider."""
    recording = RecordingClient(client)
    logical_time = [0.0]
    runtime = VehicleMindRuntime(
        recording,
        max_tool_rounds=max_tool_rounds,
        quality_clock=lambda: logical_time[0],
        action_clock=lambda: logical_time[0],
    )
    schema_hash = hashlib.sha256(
        json.dumps(runtime.tools.llm_schemas(), sort_keys=True).encode()
    ).hexdigest()
    replies: list[str] = []
    interaction_events: list[dict[str, Any]] = []
    error = None
    started = time.perf_counter()
    try:
        for step in case.steps:
            if "at_ms" in step:
                logical_time[0] = step["at_ms"] / 1000
            if "cabin" in step:
                runtime.update_cabin(at_ms=step.get("at_ms"), **step["cabin"])
            if "road" in step:
                runtime.update_driving(at_ms=step.get("at_ms"), **step["road"])
            if "road_quality" in step:
                runtime.update_driving(
                    metadata=ObservationMetadata(
                        timestamp_ms=step.get("at_ms", 0),
                        sequence=0,
                        source="agent_eval",
                        confidence=None,
                        valid=False,
                        processing_ms=0,
                    ),
                    at_ms=step.get("at_ms"),
                )
            if "tool_failure" in step:
                failure = step["tool_failure"]
                runtime.tools.get(failure["name"]).handler = (
                    lambda _error=failure["error"], **_kwargs: ToolResult(
                        success=False,
                        message="Simulated tool failure.",
                        error=_error,
                    )
                )
            if "vehicle" in step:
                vehicle = dict(step["vehicle"])
                if "gear" in vehicle:
                    vehicle["gear"] = GearState(vehicle["gear"])
                if "navigation_state" in vehicle:
                    vehicle["navigation_state"] = NavigationState(
                        vehicle["navigation_state"]
                    )
                runtime.update_vehicle(**vehicle)
            if "user_text" in step:
                reply = runtime.chat(step["user_text"], debug=False)
                replies.append(reply)
                interaction_events.append(
                    {
                        "kind": "agent_reply",
                        "at_ms": step.get("at_ms"),
                        "text": reply,
                    }
                )
            if step.get("confirm_pending"):
                pending = runtime.agent.pending_actions.get()
                if pending is not None:
                    confirmation = runtime.agent.confirm_pending(pending.action_id)
                    interaction_events.append(
                        {
                            "kind": "confirmation",
                            "at_ms": step.get("at_ms"),
                            "success": confirmation.success,
                            "error": confirmation.error,
                            "result": plain_value(confirmation.to_dict()),
                        }
                    )
                else:
                    interaction_events.append(
                        {
                            "kind": "confirmation",
                            "at_ms": step.get("at_ms"),
                            "success": False,
                            "error": "NO_PENDING_ACTION",
                        }
                    )
            if step.get("reject_pending"):
                pending = runtime.agent.pending_actions.get()
                if pending is not None:
                    rejection = runtime.agent.reject_pending(pending.action_id)
                    interaction_events.append(
                        {
                            "kind": "rejection",
                            "at_ms": step.get("at_ms"),
                            "success": rejection.success,
                            "error": rejection.error,
                        }
                    )
        if any(not reply.strip() for reply in replies):
            error = "EmptyResponse"
    except Exception as exc:
        error = type(exc).__name__
    elapsed = (time.perf_counter() - started) * 1000
    tools = tuple(
        {
            "name": item.name,
            "arguments": plain_value(item.arguments),
            "success": item.success,
            "error": item.error,
            "confirmed": item.confirmed,
            "result_data": plain_value(item.result_data),
        }
        for item in runtime.tools.execution_history()
    )
    return TrialResult(
        case_id=case.id,
        provider=provider,
        model=model,
        trial_index=trial_index,
        replies=tuple(replies),
        tool_calls=tools,
        final_context=plain_value(runtime.context_manager.get_agent_context()),
        request_count=len(recording.requests),
        latency_ms=elapsed,
        error=error,
        model_responses=tuple(
            {
                "response_model": response.response_model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ],
                "content": response.content,
            }
            for response in recording.responses
        ),
        case_sha256=case.sha256,
        tool_schema_sha256=schema_hash,
        request_latencies_ms=tuple(recording.latencies_ms),
        requested_tools=tuple(
            {"name": call.name, "arguments": plain_value(call.arguments)}
            for response in recording.responses
            for call in response.tool_calls
        ),
        requests=tuple(recording.requests),
        settings={
            "temperature": getattr(client, "temperature", None),
            "timeout_seconds": getattr(client, "timeout_seconds", None),
            "max_tool_rounds": max_tool_rounds,
            "client_max_retries": 0,
        },
        interaction_events=tuple(interaction_events),
    )

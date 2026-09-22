from __future__ import annotations

import time

from collections.abc import Mapping
from typing import Any

from modules.vehicle_ai.context import GearState, NavigationState
from modules.vehicle_ai.replay.models import ReplayObservation, ReplayScenario
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.replay.trace import (
    CONTEXT_SCHEMA_VERSION,
    AssertionResult,
    ReplayResult,
    TraceRecorder,
    frozen_plain_mapping,
    plain_value,
    semantic_digest,
)
from modules.vehicle_ai.runtime import VehicleMindRuntime


class ReplayRunner:
    """Apply a validated semantic scenario to one shared VehicleMind runtime."""

    def _record_observation(
        self,
        recorder: TraceRecorder,
        *,
        at_ms: int,
        domain: str,
        observation: ReplayObservation,
        applied: bool,
    ) -> None:
        recorder.add(
            at_ms=at_ms,
            kind="context_update",
            data={
                "domain": domain,
                "source": observation.source,
                "confidence": observation.confidence,
                "valid": observation.valid,
                "applied": applied,
                "values": plain_value(observation.values),
            },
        )

    def _record_events(
        self,
        recorder: TraceRecorder,
        *,
        at_ms: int,
        events: list[Any],
    ) -> None:
        for event in events:
            recorder.add(
                at_ms=at_ms,
                kind="event",
                data={
                    "type": plain_value(event.type),
                    "priority": plain_value(event.priority),
                    "source": event.source,
                    "message": event.message,
                    "data": plain_value(event.data),
                },
            )

    def _vehicle_values(self, values: Mapping[str, object]) -> dict[str, object]:
        normalized = dict(values)
        if "gear" in normalized:
            normalized["gear"] = GearState(str(normalized["gear"]))
        if "navigation_state" in normalized:
            normalized["navigation_state"] = NavigationState(
                str(normalized["navigation_state"])
            )
        return normalized

    def _apply_observation(
        self,
        runtime: VehicleMindRuntime,
        recorder: TraceRecorder,
        *,
        at_ms: int,
        domain: str,
        observation: ReplayObservation,
    ) -> None:
        self._record_observation(
            recorder,
            at_ms=at_ms,
            domain=domain,
            observation=observation,
            applied=observation.valid,
        )
        if not observation.valid:
            return
        values = dict(observation.values)
        if domain == "cabin":
            events = runtime.update_cabin(**values)
        elif domain == "road":
            events = runtime.update_driving(**values)
        else:
            events = runtime.update_vehicle(**self._vehicle_values(values))
        self._record_events(recorder, at_ms=at_ms, events=events)

    def _record_new_tools(
        self,
        runtime: VehicleMindRuntime,
        recorder: TraceRecorder,
        *,
        at_ms: int,
        start_index: int,
    ) -> int:
        history = runtime.tools.execution_history()
        for item in history[start_index:]:
            recorder.add(
                at_ms=at_ms,
                kind="tool_result",
                data={
                    "name": item.name,
                    "arguments": plain_value(item.arguments),
                    "requires_confirmation": item.requires_confirmation,
                    "confirmed": item.confirmed,
                    "success": item.success,
                    "error": item.error,
                },
            )
        return len(history)

    def _assertions(
        self,
        scenario: ReplayScenario,
        *,
        event_types: tuple[str, ...],
        successful_tools: tuple[str, ...],
        final_context: Mapping[str, object],
        unauthorized: int,
        remaining_responses: int,
    ) -> tuple[AssertionResult, ...]:
        assertions: list[AssertionResult] = []
        for event_type in scenario.expected.event_types:
            assertions.append(
                AssertionResult(
                    name=f"event:{event_type}",
                    passed=event_type in event_types,
                    expected=True,
                    actual=event_type in event_types,
                )
            )
        for tool_name in scenario.expected.successful_tools:
            assertions.append(
                AssertionResult(
                    name=f"tool:{tool_name}",
                    passed=tool_name in successful_tools,
                    expected=True,
                    actual=tool_name in successful_tools,
                )
            )
        vehicle = final_context["vehicle"]
        assert isinstance(vehicle, Mapping)
        for field, expected in scenario.expected.final_vehicle.items():
            actual = vehicle.get(field)
            assertions.append(
                AssertionResult(
                    name=f"vehicle.{field}",
                    passed=actual == expected,
                    expected=expected,
                    actual=actual,
                )
            )
        assertions.extend(
            [
                AssertionResult(
                    name="unauthorized_sensitive_executions",
                    passed=(
                        unauthorized
                        == scenario.expected.unauthorized_sensitive_executions
                    ),
                    expected=scenario.expected.unauthorized_sensitive_executions,
                    actual=unauthorized,
                ),
                AssertionResult(
                    name="scripted_responses_consumed",
                    passed=remaining_responses == 0,
                    expected=0,
                    actual=remaining_responses,
                ),
            ]
        )
        return tuple(assertions)

    def run(self, scenario: ReplayScenario) -> ReplayResult:
        started = time.perf_counter()
        llm = ScriptedLLMClient(scenario.responses)
        runtime = VehicleMindRuntime(llm=llm)
        recorder = TraceRecorder()
        tool_index = 0

        for step in scenario.steps:
            for domain in ("vehicle", "cabin", "road"):
                observation = getattr(step, domain)
                if observation is not None:
                    self._apply_observation(
                        runtime,
                        recorder,
                        at_ms=step.at_ms,
                        domain=domain,
                        observation=observation,
                    )

            if step.user_text is not None:
                recorder.add(
                    at_ms=step.at_ms,
                    kind="user_utterance",
                    data={"text": step.user_text},
                )
                answer = runtime.chat(step.user_text, debug=False)
                recorder.add(
                    at_ms=step.at_ms,
                    kind="agent_response",
                    data={"summary": answer},
                )
                pending = runtime.agent.pending_actions.get()
                if pending is not None:
                    recorder.add(
                        at_ms=step.at_ms,
                        kind="pending_action",
                        data={
                            "tool_name": pending.tool_name,
                            "arguments": plain_value(pending.arguments),
                            "display_text": pending.display_text,
                        },
                    )
                tool_index = self._record_new_tools(
                    runtime,
                    recorder,
                    at_ms=step.at_ms,
                    start_index=tool_index,
                )

            if step.confirm_pending:
                pending = runtime.agent.pending_actions.get()
                if pending is None:
                    raise RuntimeError(
                        "scenario requested confirmation without an action"
                    )
                result = runtime.agent.confirm_pending(pending.action_id)
                recorder.add(
                    at_ms=step.at_ms,
                    kind="confirmation",
                    data={
                        "tool_name": pending.tool_name,
                        "arguments": plain_value(pending.arguments),
                        "success": result.success,
                        "error": result.error,
                    },
                )
                tool_index = self._record_new_tools(
                    runtime,
                    recorder,
                    at_ms=step.at_ms,
                    start_index=tool_index,
                )

        context = frozen_plain_mapping(
            runtime.context_manager.get_context().to_agent_context()
        )
        history = runtime.tools.execution_history()
        event_types = tuple(
            str(record.data["type"])
            for record in recorder.records()
            if record.kind == "event"
        )
        successful_tools = tuple(item.name for item in history if item.success)
        unauthorized = sum(
            1
            for item in history
            if item.requires_confirmation and item.success and not item.confirmed
        )
        assertions = self._assertions(
            scenario,
            event_types=event_types,
            successful_tools=successful_tools,
            final_context=context,
            unauthorized=unauthorized,
            remaining_responses=llm.remaining,
        )
        final_at_ms = scenario.steps[-1].at_ms
        for item in assertions:
            recorder.add(
                at_ms=final_at_ms,
                kind="assertion",
                data={
                    "name": item.name,
                    "passed": item.passed,
                    "expected": plain_value(item.expected),
                    "actual": plain_value(item.actual),
                },
            )
        trace = recorder.records()
        return ReplayResult(
            context_schema_version=CONTEXT_SCHEMA_VERSION,
            scenario_id=scenario.scenario_id,
            passed=all(item.passed for item in assertions),
            trace=trace,
            semantic_sha256=semantic_digest(scenario.scenario_id, trace),
            assertions=assertions,
            final_context=context,
            metrics={"total_ms": (time.perf_counter() - started) * 1000},
            event_types=event_types,
            successful_tools=successful_tools,
            unauthorized_sensitive_executions=unauthorized,
            remaining_scripted_responses=llm.remaining,
        )

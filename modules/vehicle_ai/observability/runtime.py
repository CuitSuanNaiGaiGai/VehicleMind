from __future__ import annotations

from typing import Any

from modules.vehicle_ai.observability.llm import TracingLLMClient
from modules.vehicle_ai.observability.trace import AgentTraceRecorder, TraceBackend
from modules.vehicle_ai.workflow import VehicleAgentWorkflow


class RuntimeTracing:
    """Bind one recorder to a runtime's agent and tool authorization boundary."""

    def __init__(
        self,
        owner: Any,
        thread_id: str,
        backend: TraceBackend | None = None,
    ) -> None:
        self.owner = owner
        self.recorder = AgentTraceRecorder(
            thread_id,
            backend=backend,
            task_id_provider=self._current_task_id,
        )

    def _current_task_id(self) -> str:
        agent = getattr(self.owner, "agent", None)
        return agent.task.task_id if agent is not None else "unassigned"

    def configure_workflow(self, agent: Any, tools: Any, thread_id: str):
        agent.llm = TracingLLMClient(agent.llm, self.recorder)
        agent.trace_recorder = self.recorder
        agent.confirmations.trace_recorder = self.recorder
        tools.set_trace_recorder(self.recorder)
        return VehicleAgentWorkflow(
            agent,
            thread_id=thread_id,
            trace_recorder=self.recorder,
        )

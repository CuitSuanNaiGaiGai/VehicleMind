"""Shared deterministic fixtures for business-dialogue flow tests."""

import json

from modules.config.agent_dialogue import AgentDialogueConfig
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime


def interpretation(intent, *, changes=(), reference=None, evidence="找附近服务区"):
    return ScriptedResponse(
        content=json.dumps(
            {
                "intent": intent,
                "changes": list(changes),
                "reference": reference,
                "evidence": evidence,
                "clarification_reason": None,
            },
            ensure_ascii=False,
        )
    )


def make_runtime(*responses, clock=None, **kwargs):
    now = clock if clock is not None else [0.0]
    runtime = VehicleMindRuntime(
        ScriptedLLMClient(tuple(responses)),
        dialogue_config=AgentDialogueConfig(enabled=True),
        action_clock=lambda: now[0],
        quality_clock=lambda: now[0],
        enable_event_recommendations=False,
        **kwargs,
    )
    return runtime, now

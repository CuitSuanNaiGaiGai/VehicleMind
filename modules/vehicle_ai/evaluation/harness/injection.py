"""Deterministic adversarial gate checks shared by pytest and promptfoo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from modules.vehicle_ai.knowledge.models import (
    KnowledgeReference,
    RetrievedChunk,
    RetrievalResult,
)
from modules.vehicle_ai.llm.base import LLMToolCall
from modules.vehicle_ai.replay.models import ScriptedResponse
from modules.vehicle_ai.replay.scripted_llm import ScriptedLLMClient
from modules.vehicle_ai.runtime import VehicleMindRuntime

DEFAULT_SECURITY_SUITE = "scenarios/agent_eval/security/prompt_injection.yaml"


def load_security_suite(path: Path) -> dict[str, Any]:
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1:
        raise ValueError("unsupported security suite schema")
    if suite.get("evaluation_mode") != "deterministic_adversarial_tool_attempt":
        raise ValueError("unsupported security suite mode")
    return suite


def _tool_call(name: str, arguments: dict[str, Any] | None = None) -> ScriptedResponse:
    arguments = arguments or {}
    return ScriptedResponse(
        content=None,
        tool_calls=(
            LLMToolCall(
                f"injection-{name}",
                name,
                arguments,
                json.dumps(arguments, ensure_ascii=False),
            ),
        ),
    )


class _InjectedKnowledge:
    """Retrieval stub that returns one untrusted adversarial chunk."""

    def __init__(self, evidence: str):
        self.evidence = evidence

    def query(self, profile, query, *, top_k=5):
        reference = KnowledgeReference(
            source_id="untrusted-note",
            title="待审查的检索片段",
            section="摘要",
            source_uri="local://injection-test",
            profile=profile or "vehicle_common",
        )
        return RetrievalResult(
            profile=profile or "vehicle_common",
            query=query,
            chunks=(
                RetrievedChunk(
                    source_id=reference.source_id,
                    text=self.evidence,
                    rank=1,
                    reference=reference,
                ),
            ),
            latency_ms=0.0,
            request_id="injection-test",
        )


def run_injection_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one adversarial case against the execution-layer confirmation gate.

    The scripted client deliberately issues the sensitive tool request to
    simulate a model that complied with the injected instruction. The gate
    must intercept it without prior user confirmation.
    """
    responses = [_tool_call("search_nearby_rest_area")]
    if case["source"] == "retrieved_knowledge":
        responses.append(_tool_call("search_vehicle_knowledge", {"query": "休息建议"}))
    responses.append(_tool_call("start_navigation", {"poi_id": "rest_area_001"}))
    knowledge_client = None
    if case["source"] == "retrieved_knowledge":
        knowledge_client = _InjectedKnowledge(case["evidence"])
    runtime = VehicleMindRuntime(
        llm=ScriptedLLMClient(tuple(responses)),
        knowledge_profile="vehicle_common" if knowledge_client else None,
        knowledge_client=knowledge_client,
    )

    reply = runtime.chat(case["user_text"], debug=False)
    attempts = runtime.tools.execution_history()
    navigation_attempt = next(
        item for item in attempts if item.name == case["expected"]["tool"]
    )
    pending = runtime.agent.pending_actions.get()
    return {
        "case_id": case["id"],
        "source": case["source"],
        "reply": reply,
        "navigation_state": runtime.context_manager.get_context().vehicle.navigation_state.value,
        "error": navigation_attempt.error,
        "confirmed": navigation_attempt.confirmed,
        "success": navigation_attempt.success,
        "pending_action": pending is not None,
        "unconfirmed_sensitive_writes": sum(
            item.name == "start_navigation" and item.success and not item.confirmed
            for item in attempts
        ),
    }


def injection_gate_case(case_id: str, suite_path: Path) -> dict[str, Any]:
    suite = load_security_suite(suite_path)
    for case in suite["cases"]:
        if case["id"] == case_id:
            return run_injection_case(case)
    raise KeyError(f"unknown injection case: {case_id}")

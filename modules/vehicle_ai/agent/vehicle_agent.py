from __future__ import annotations

import json
import time
from collections.abc import Callable

from modules.vehicle_ai.agent.action_state import (
    PendingAction,
    PendingActionStore,
)
from modules.vehicle_ai.agent.confirmation import ActionConfirmationController
from modules.vehicle_ai.agent.task_state import AgentTask, TaskStatus
from modules.vehicle_ai.agent.turn import run_turn, reconcile
from modules.vehicle_ai.agent.session import (
    evidence_message,
    is_pointer,
    record,
    record_final_response,
    trip_memory_message,
)
from modules.vehicle_ai.agent.decision_context import (
    GROUNDING_NOTE,
    attach_field_evidence,
)
from modules.vehicle_ai.agent.event_advice import build_event_advice_messages
from modules.vehicle_ai.agent.pending_intent import (
    classify_pending_intent,
    requested_target,
    search_result_matches_target,
)

from modules.vehicle_ai.agent.prompts import (
    SYSTEM_PROMPT,
)
from modules.vehicle_ai.agent.message_helpers import (
    assistant_tool_message,
    pending_action_message,
    print_pending_action,
)

from modules.vehicle_ai.context import (
    ContextManager,
    ContextSelector,
)
from modules.vehicle_ai.events import VehicleEvent

from modules.vehicle_ai.llm import (
    BaseLLMClient,
)

from modules.vehicle_ai.tools import (
    ToolRegistry,
    ToolResult,
)


class VehicleAgent:
    """Context-aware agent with grounded tools and an observable task lifecycle."""

    def __init__(
        self,
        llm: BaseLLMClient,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        max_tool_rounds: int = 5,
        action_clock: Callable[[], float] = time.time,
        turn_timeout_seconds: float = 90.0,
        max_tool_calls: int = 10,
        budget_clock: Callable[[], float] = time.monotonic,
        max_task_trace_events: int = 200,
        trip_id: str | None = None,
        historical_event_sink: Callable[[dict], None] | None = None,
    ):
        self.llm = llm

        self.context_manager = context_manager

        self.tool_registry = tool_registry
        self.current_user_intent = ""
        self._turn_music_warning = ""
        self._turn_music_paused = False

        self.max_tool_rounds = max_tool_rounds
        self.turn_timeout_seconds = turn_timeout_seconds
        self.max_tool_calls = max_tool_calls
        self.budget_clock = budget_clock
        if max_task_trace_events < 1:
            raise ValueError("max_task_trace_events must be positive")
        self.max_task_trace_events = max_task_trace_events

        self.history: list[dict] = []
        self.task = AgentTask()
        self.trace: list[dict] = []
        self.trace_sequence = 0

        self.pending_actions = PendingActionStore(clock=action_clock)
        self.confirmations = ActionConfirmationController(
            self.pending_actions,
            self.tool_registry,
            self.tool_registry.take_confirmation_issuer(),
        )
        self.context_selector = ContextSelector()
        self.trip_id = trip_id
        self.historical_event_sink = historical_event_sink

    def _context_message(
        self,
        user_text: str,
        debug: bool = False,
    ) -> dict:

        # --------------------------------------------------------
        # Obtain latest complete runtime context.
        # --------------------------------------------------------

        full_context = self.context_manager.get_context()

        # --------------------------------------------------------
        # Select only information relevant to this turn.
        # --------------------------------------------------------

        selection = self.context_selector.select(
            user_text=(user_text),
            vehicle_context=(full_context),
            road_quality=self.context_manager.observation_quality("road")["status"],
            driver_quality=self.context_manager.observation_quality("driver")["status"],
            vehicle_quality=self.context_manager.observation_quality("vehicle")[
                "status"
            ],
            field_quality=self.context_manager.field_quality,
        )

        selected_context = attach_field_evidence(
            selection.context, self.context_manager
        )

        if debug:
            print()
            print("[Context Selector]")

            print(
                "  topics:",
                [topic.value for topic in selection.topics],
            )

            print(
                "  matched:",
                selection.matched_keywords,
            )

            print("  context:")

            print(
                json.dumps(
                    selected_context,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

        # --------------------------------------------------------
        # Nothing relevant.
        # --------------------------------------------------------

        if not selected_context:
            return {
                "role": "system",
                "content": (
                    "CURRENT RELEVANT "
                    "VEHICLE CONTEXT:\n"
                    "No vehicle context is "
                    "required for this request."
                ),
            }

        context_json = json.dumps(
            selected_context,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        return {
            "role": "system",
            "content": (
                "CURRENT RELEVANT "
                "VEHICLE CONTEXT:\n"
                f"{context_json}\n\n"
                "Only use this context when "
                "it is relevant to the "
                "current user request. "
                "If a domain quality is STALE, INVALID or MISSING, "
                "prior turns and stored values do not establish current facts for that domain."
                f"\n{GROUNDING_NOTE}"
            ),
        }

    def _pending_action_message(
        self,
    ) -> dict:
        return pending_action_message(self.pending_actions)

    def _assistant_tool_message(
        self,
        response,
    ) -> dict:
        return assistant_tool_message(response)

    def _ground_tool_arguments(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """Use the pending navigation's canonical POI to prevent entity drift."""

        grounded = dict(arguments)

        if tool_name == "start_navigation":
            pending = self.pending_actions.get()

            if pending is not None and pending.tool_name == "start_navigation":
                pending_poi_id = pending.arguments.get("poi_id")

                if pending_poi_id:
                    grounded["poi_id"] = pending_poi_id

        return grounded

    def _update_action_state(
        self,
        tool_name: str,
        tool_result,
        target: str | None = None,
    ) -> None:
        """Convert search observations into actions awaiting driver confirmation."""

        if not tool_result.success:
            return

        if tool_name == "search_nearby_rest_area":
            if target is not None and not search_result_matches_target(
                target, tool_result.data
            ):
                return
            poi_id = tool_result.data.get("poi_id")

            name = tool_result.data.get("name")

            if poi_id and name:
                self.pending_actions.set(
                    PendingAction(
                        tool_name=("start_navigation"),
                        arguments={
                            "poi_id": poi_id,
                        },
                        display_text=(f"Navigate to {name}"),
                        metadata={
                            "poi_id": poi_id,
                            "name": name,
                            "distance_km": tool_result.data.get("distance_km"),
                            "eta_minutes": tool_result.data.get("eta_minutes"),
                            "user_intent": self.current_user_intent,
                        },
                        expires_after_seconds=(120.0),
                        created_at=self.pending_actions.now(),
                    )
                )

        elif tool_name == "start_navigation":
            self.pending_actions.clear()

        elif tool_name == "cancel_navigation":
            self.pending_actions.clear()

    def confirm_pending(self, action_id: str) -> ToolResult:
        pending = self.pending_actions.get()
        action = pending.to_dict() if pending and pending.action_id == action_id else None
        result = self.confirmations.confirm(action_id)
        if result.error != "INVALID_CONFIRMATION":
            self.task.last_tool_result = result.to_dict()
            self.task.tool_results.append(result.to_dict())
            self.task.transition(
                TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                result.error,
            )
            if result.data.get("outcome_unknown"):
                reconcile(
                    self,
                    self.task.pending_action.get("tool_name", "")
                    if self.task.pending_action
                    else "",
                    result,
                )
        elif (
            self.task.status is TaskStatus.AWAITING_CONFIRMATION
            and self.pending_actions.get() is None
        ):
            self.task.transition(TaskStatus.AWAITING_INPUT, "PENDING_EXPIRED")
        record(
            self,
            "confirmation",
            source="confirmation_controller",
            quality="TOOL_RESULT",
            action=action,
            result=result.to_dict(),
        )
        return result

    def reject_pending(self, action_id: str) -> ToolResult:
        pending = self.pending_actions.get()
        action = pending.to_dict() if pending and pending.action_id == action_id else None
        result = self.confirmations.reject(action_id)
        if result.success:
            self.task.transition(TaskStatus.CANCELLED, "USER_CANCELLED")
        record(
            self,
            "rejection",
            source="confirmation_controller",
            quality="TOOL_RESULT",
            action=action,
            result=result.to_dict(),
        )
        return result

    def _print_pending_action(
        self,
    ) -> None:
        print_pending_action(self.pending_actions)

    # ========================================================
    # Chat
    # ========================================================

    def _record_final_response(self, user_text: str, answer: str) -> str:
        return record_final_response(self, user_text, answer)

    def chat(
        self,
        user_text: str,
        debug: bool = True,
    ) -> str:

        user_text = user_text.strip()

        if not user_text:
            return ""
        self.current_user_intent = user_text
        self._turn_music_warning = ""
        self._turn_music_paused = False

        pending = self.pending_actions.get()
        expired = self.task.pending_action is not None and pending is None
        if expired:
            self.task.transition(TaskStatus.AWAITING_INPUT, "PENDING_EXPIRED")
            record(self, "pending_expired", source="pending_store", quality="EXPIRED")
        if is_pointer(user_text) and pending is None:
            self.task.transition(
                TaskStatus.AWAITING_INPUT,
                "PENDING_EXPIRED" if expired else "NO_REFERENT",
            )
            record(
                self,
                "user_request",
                source="user",
                quality="SELF_REPORTED",
                text=user_text,
            )
            return self._record_final_response(
                user_text,
                "待确认操作已过期，请重新选择目标。"
                if expired
                else "目前没有可确认的目标，请说明要执行什么操作。",
            )
        if pending is None and self.task.status is TaskStatus.AWAITING_INPUT:
            previous_topics, previous_matches = self.context_selector.detect_topics(
                self.task.goal
            )
            current_topics, current_matches = self.context_selector.detect_topics(
                user_text
            )
            previous_primary = set(previous_matches)
            current_primary = set(current_matches)
            if (
                previous_topics
                and current_topics
                and previous_primary.isdisjoint(current_primary)
            ):
                self.task = AgentTask(goal=user_text)
        elif pending is None and self.task.status is not TaskStatus.AWAITING_INPUT:
            self.task = AgentTask(goal=user_text)
        elif expired and not is_pointer(user_text):
            self.task = AgentTask(goal=user_text)
        self.task.transition(TaskStatus.RUNNING)
        record(
            self, "user_request", source="user", quality="SELF_REPORTED", text=user_text
        )
        intent = classify_pending_intent(user_text)
        if pending is not None and intent in {"reject", "change_target"}:
            previous_action = pending.to_dict()
            if not self.confirmations.reject(pending.action_id).success:
                self.task.transition(TaskStatus.AWAITING_INPUT, "PENDING_CHANGED")
                return self._record_final_response(
                    user_text, "待确认操作已变更，请重新确认当前操作。"
                )
            if intent == "change_target":
                record(
                    self,
                    "selection",
                    source="user",
                    quality="SELF_REPORTED",
                    action=previous_action,
                    text=user_text,
                    selected_target=requested_target(user_text),
                )
        if intent == "reject":
            self.task.transition(TaskStatus.CANCELLED, "USER_CANCELLED")
            record(
                self,
                "rejection",
                source="user",
                quality="SELF_REPORTED",
                action=previous_action if pending is not None else None,
                result={"success": True, "message": "用户取消待确认操作。"},
            )
            return self._record_final_response(user_text, "已取消待确认操作。")
        target = requested_target(user_text) if intent == "change_target" else None
        if intent == "change_target":
            self.task.goal = user_text
            self.task.transition(TaskStatus.RUNNING, "TARGET_CHANGED")

        # ----------------------------------------------------
        # Rebuild dynamic system state every user turn.
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            self._context_message(
                user_text=f"{self.task.goal} {user_text}",
                debug=debug,
            ),
            self._pending_action_message(),
            evidence_message(self),
            *self.history,
            {
                "role": "user",
                "content": user_text,
            },
        ]
        memory_message = trip_memory_message(self.tool_registry.names())
        if memory_message is not None:
            messages.insert(3, memory_message)

        return run_turn(self, user_text, messages, target)

    # ========================================================
    # Reset
    # ========================================================

    def recommend_from_event(self, event: VehicleEvent) -> str:
        """Generate one advisory reply without entering the task or tool workflow."""
        messages, evidence = build_event_advice_messages(event)
        response = self.llm.chat_with_timeout(
            messages, tools=[], timeout_seconds=self.turn_timeout_seconds
        )
        text = response.content or ""
        self.trace.append(
            {
                "sequence": self.trace_sequence,
                "kind": "event_recommendation",
                "event_id": event.event_id,
                "event_time": event.timestamp,
                "evidence": evidence,
                "historical_context": event.data.get("prior_trip_interactions", []),
                "text": text,
                "tool_calls_ignored": len(response.tool_calls),
            }
        )
        self.trace_sequence += 1
        del self.trace[: -self.max_task_trace_events]
        return text

    def reset(
        self,
    ) -> None:

        self.history.clear()

        self.pending_actions.clear()
        self.task = AgentTask()
        self.trace.clear()
        self.trace_sequence = 0
        self.current_user_intent = ""
        self._turn_music_warning = ""
        self._turn_music_paused = False

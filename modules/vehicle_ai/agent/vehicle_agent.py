from __future__ import annotations

import json

from modules.vehicle_ai.agent.action_state import (
    PendingAction,
    PendingActionStore,
)
from modules.vehicle_ai.agent.confirmation import ActionConfirmationController

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

from modules.vehicle_ai.llm import (
    BaseLLMClient,
)

from modules.vehicle_ai.tools import (
    ToolRegistry,
    ToolResult,
)


class VehicleAgent:
    """
    Context-aware VehicleMind Agent.

    Core flow:

        User
          ↓
        Vehicle Context
          +
        Pending Action
          ↓
        LLM
          ↓
        Function Call
          ↓
        ToolRegistry
          ↓
        Tool Result
          ↓
        Action State Update
          ↓
        LLM Final Response
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        max_tool_rounds: int = 5,
    ):
        self.llm = llm

        self.context_manager = context_manager

        self.tool_registry = tool_registry

        self.max_tool_rounds = max_tool_rounds

        # ----------------------------------------------------
        # Natural language history
        # ----------------------------------------------------

        self.history: list[dict] = []

        # ----------------------------------------------------
        # Grounded action state
        # ----------------------------------------------------

        self.pending_actions = PendingActionStore()
        self.confirmations = ActionConfirmationController(
            self.pending_actions,
            self.tool_registry,
        )
        self.context_selector = ContextSelector()

    # ========================================================
    # Vehicle Context
    # ========================================================

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
        )

        selected_context = selection.context

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
                "current user request."
            ),
        }

    # ========================================================
    # Pending Action Context
    # ========================================================

    def _pending_action_message(
        self,
    ) -> dict:
        return pending_action_message(self.pending_actions)

    # ========================================================
    # Tool-call assistant message
    # ========================================================

    def _assistant_tool_message(
        self,
        response,
    ) -> dict:
        return assistant_tool_message(response)

    # ========================================================
    # Tool argument grounding
    # ========================================================

    def _ground_tool_arguments(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        """
        Apply deterministic grounding before tool execution.

        For start_navigation:

        if a valid pending navigation action exists, the
        canonical poi_id from the pending action is authoritative.

        This prevents entity drift caused by LLM paraphrasing.
        """

        grounded = dict(arguments)

        if tool_name == "start_navigation":
            pending = self.pending_actions.get()

            if pending is not None and pending.tool_name == "start_navigation":
                pending_poi_id = pending.arguments.get("poi_id")

                if pending_poi_id:
                    grounded["poi_id"] = pending_poi_id

        return grounded

    # ========================================================
    # Tool result -> Action State
    # ========================================================

    def _update_action_state(
        self,
        tool_name: str,
        tool_result,
    ) -> None:
        """
        Convert selected tool results into grounded pending
        actions.

        Search results are observations.

        Actions derived from those observations are stored
        separately until confirmed by the driver.
        """

        if not tool_result.success:
            return

        # ----------------------------------------------------
        # Rest-area search
        # ----------------------------------------------------

        if tool_name == "search_nearby_rest_area":
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
                        },
                        expires_after_seconds=(120.0),
                    )
                )

        # ----------------------------------------------------
        # Navigation started
        # ----------------------------------------------------

        elif tool_name == "start_navigation":
            self.pending_actions.clear()

        # ----------------------------------------------------
        # Navigation cancelled
        # ----------------------------------------------------

        elif tool_name == "cancel_navigation":
            self.pending_actions.clear()

    def confirm_pending(self, action_id: str) -> ToolResult:
        return self.confirmations.confirm(action_id)

    # ========================================================
    # Debug
    # ========================================================

    def _print_pending_action(
        self,
    ) -> None:
        print_pending_action(self.pending_actions)

    # ========================================================
    # Chat
    # ========================================================

    def chat(
        self,
        user_text: str,
        debug: bool = True,
    ) -> str:

        user_text = user_text.strip()

        if not user_text:
            return ""

        # ----------------------------------------------------
        # Rebuild dynamic system state every user turn.
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            self._context_message(
                user_text=user_text,
                debug=debug,
            ),
            self._pending_action_message(),
            *self.history,
            {
                "role": "user",
                "content": user_text,
            },
        ]

        tools = self.tool_registry.llm_schemas()

        # ====================================================
        # Agent Loop
        # ====================================================

        for _ in range(self.max_tool_rounds):
            response = self.llm.chat(
                messages=messages,
                tools=tools,
            )

            # =================================================
            # Final natural-language response
            # =================================================

            if not response.tool_calls:
                final_text = response.content or ""

                self.history.append(
                    {
                        "role": "user",
                        "content": user_text,
                    }
                )

                self.history.append(
                    {
                        "role": "assistant",
                        "content": final_text,
                    }
                )

                self.history = self.history[-12:]

                return final_text

            # =================================================
            # Tool Calls
            # =================================================

            messages.append(self._assistant_tool_message(response))

            for call in response.tool_calls:
                # --------------------------------------------
                # Deterministic grounding
                # --------------------------------------------

                grounded_arguments = self._ground_tool_arguments(
                    tool_name=(call.name),
                    arguments=(call.arguments),
                )
                self.confirmations.stage(call.name, grounded_arguments)

                if debug:
                    print()
                    print("[Agent Tool Call]")

                    print(f"  {call.name}")

                    print(
                        json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )

                    if grounded_arguments != call.arguments:
                        print("[Grounded Arguments]")

                        print(
                            json.dumps(
                                grounded_arguments,
                                ensure_ascii=False,
                                indent=2,
                            )
                        )

                # --------------------------------------------
                # Execute
                # --------------------------------------------

                tool_result = self.tool_registry.execute(
                    name=(call.name),
                    arguments=(grounded_arguments),
                )

                # --------------------------------------------
                # Update deterministic action state
                # --------------------------------------------

                self._update_action_state(
                    tool_name=(call.name),
                    tool_result=(tool_result),
                )

                if debug:
                    print("[Tool Result]")

                    print(
                        json.dumps(
                            tool_result.to_dict(),
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    )

                    self._print_pending_action()

                result_json = json.dumps(
                    tool_result.to_dict(),
                    ensure_ascii=False,
                    default=str,
                )

                # --------------------------------------------
                # Tool response
                # --------------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_json,
                    }
                )

        return "本次请求涉及过多连续工具调用，已停止执行。"

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.history.clear()

        self.pending_actions.clear()

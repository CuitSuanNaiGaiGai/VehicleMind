from __future__ import annotations

import json

from modules.vehicle_ai.context import (
    ContextManager,
)

from modules.vehicle_ai.llm import (
    BaseLLMClient,
)

from modules.vehicle_ai.tools import (
    ToolRegistry,
)

from modules.vehicle_ai.agent.prompts import (
    SYSTEM_PROMPT,
)


class VehicleAgent:
    """
    Context-aware VehicleMind Agent.

    Flow:

        User
          ↓
        Current VehicleContext
          ↓
        LLM
          ↓
        tool_calls?
          │
       ┌──┴──┐
       NO    YES
       │      │
       ▼      ▼
    answer   ToolRegistry
              │
              ▼
           ToolResult
              │
              ▼
             LLM
              │
              ▼
         final answer
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        max_tool_rounds: int = 5,
    ):
        self.llm = llm

        self.context_manager = (
            context_manager
        )

        self.tool_registry = (
            tool_registry
        )

        self.max_tool_rounds = (
            max_tool_rounds
        )

        # Natural-language conversation only.
        #
        # Dynamic vehicle state is intentionally
        # re-injected from ContextManager every turn.
        self.history: list[
            dict
        ] = []

    # ========================================================
    # Context
    # ========================================================

    def _context_message(
        self,
    ) -> dict:

        context = (
            self.context_manager
            .get_agent_context()
        )

        context_json = (
            json.dumps(
                context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        return {
            "role": "system",
            "content": (
                "CURRENT VEHICLE CONTEXT:\n"
                f"{context_json}"
            ),
        }

    # ========================================================
    # Tool assistant message
    # ========================================================

    def _assistant_tool_message(
        self,
        response,
    ) -> dict:

        return {
            "role":
                "assistant",

            "content":
                response.content,

            "tool_calls": [
                {
                    "id":
                        call.id,

                    "type":
                        "function",

                    "function": {
                        "name":
                            call.name,

                        "arguments":
                            call.arguments_json,
                    },
                }
                for call
                in response.tool_calls
            ],
        }

    # ========================================================
    # Chat
    # ========================================================

    def chat(
        self,
        user_text: str,
        debug: bool = True,
    ) -> str:

        user_text = (
            user_text.strip()
        )

        if not user_text:

            return ""

        # ----------------------------------------------------
        # Rebuild dynamic context every user turn.
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content":
                    SYSTEM_PROMPT,
            },

            self._context_message(),

            *self.history,

            {
                "role": "user",
                "content": user_text,
            },
        ]

        tools = (
            self.tool_registry
            .llm_schemas()
        )

        # ====================================================
        # Agent loop
        # ====================================================

        for round_index in range(
            self.max_tool_rounds
        ):

            response = (
                self.llm.chat(
                    messages=messages,
                    tools=tools,
                )
            )

            # ------------------------------------------------
            # No tool call -> final response
            # ------------------------------------------------

            if not response.tool_calls:

                final_text = (
                    response.content
                    or ""
                )

                self.history.append(
                    {
                        "role":
                            "user",
                        "content":
                            user_text,
                    }
                )

                self.history.append(
                    {
                        "role":
                            "assistant",
                        "content":
                            final_text,
                    }
                )

                # Keep a small conversational window.
                self.history = (
                    self.history[-12:]
                )

                return final_text

            # ------------------------------------------------
            # Tool calls
            # ------------------------------------------------

            messages.append(
                self._assistant_tool_message(
                    response
                )
            )

            for call in (
                response.tool_calls
            ):

                if debug:

                    print()
                    print(
                        "[Agent Tool Call]"
                    )

                    print(
                        f"  {call.name}"
                    )

                    print(
                        json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )

                tool_result = (
                    self.tool_registry
                    .execute(
                        name=(
                            call.name
                        ),
                        arguments=(
                            call.arguments
                        ),
                    )
                )

                result_json = (
                    json.dumps(
                        tool_result
                        .to_dict(),
                        ensure_ascii=False,
                        default=str,
                    )
                )

                if debug:

                    print(
                        "[Tool Result]"
                    )

                    print(
                        json.dumps(
                            tool_result
                            .to_dict(),
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        )
                    )

                # --------------------------------------------
                # Function Calling protocol:
                #
                # assistant(tool_calls)
                #        ↓
                # tool(tool_call_id)
                #        ↓
                # assistant(final answer)
                # --------------------------------------------

                messages.append(
                    {
                        "role":
                            "tool",

                        "tool_call_id":
                            call.id,

                        "content":
                            result_json,
                    }
                )

        return (
            "本次请求涉及过多连续工具调用，"
            "已停止执行。"
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
    ) -> None:

        self.history.clear()
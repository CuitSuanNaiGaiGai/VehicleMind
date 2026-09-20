from __future__ import annotations

import json
import os

from openai import OpenAI

from modules.vehicle_ai.llm.base import (
    BaseLLMClient,
    LLMResponse,
    LLMToolCall,
)


class QwenClient(
    BaseLLMClient
):
    """
    Qwen through Alibaba Cloud Model Studio
    OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = (
            api_key
            or os.getenv(
                "DASHSCOPE_API_KEY"
            )
        )

        if not self.api_key:

            raise RuntimeError(
                "DASHSCOPE_API_KEY "
                "is not configured."
            )

        self.model = (
            model
            or os.getenv(
                "QWEN_MODEL",
                "qwen3.8-max",
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "QWEN_BASE_URL",
                (
                    "https://"
                    "dashscope.aliyuncs.com/"
                    "compatible-mode/v1"
                ),
            )
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        print(
            "[VehicleMind] "
            "LLM provider: Qwen"
        )

        print(
            "[VehicleMind] "
            f"LLM model: {self.model}"
        )

    def chat(
        self,
        messages,
        tools=None,
    ) -> LLMResponse:

        kwargs = {
            "model":
                self.model,

            "messages":
                messages,

            "temperature":
                0.2,

            # Function calling is easier to
            # maintain without preserved CoT.
            "extra_body": {
                "enable_thinking":
                    False,
            },
        }

        if tools:

            kwargs[
                "tools"
            ] = tools

            kwargs[
                "tool_choice"
            ] = "auto"

        response = (
            self.client
            .chat
            .completions
            .create(
                **kwargs
            )
        )

        choice = (
            response
            .choices[0]
        )

        message = (
            choice.message
        )

        normalized_calls = []

        for call in (
            message.tool_calls
            or []
        ):

            arguments_json = (
                call
                .function
                .arguments
                or "{}"
            )

            try:

                arguments = (
                    json.loads(
                        arguments_json
                    )
                )

            except json.JSONDecodeError:

                arguments = {}

            normalized_calls.append(
                LLMToolCall(
                    id=call.id,
                    name=(
                        call
                        .function
                        .name
                    ),
                    arguments=(
                        arguments
                    ),
                    arguments_json=(
                        arguments_json
                    ),
                )
            )

        return LLMResponse(
            content=(
                message.content
            ),
            tool_calls=(
                normalized_calls
            ),
            finish_reason=(
                choice.finish_reason
            ),
        )
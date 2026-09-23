from __future__ import annotations

import json
import os

from openai import OpenAI

from modules.vehicle_ai.llm.base import (
    BaseLLMClient,
    LLMResponse,
    LLMToolCall,
)


class QwenClient(BaseLLMClient):
    """
    Qwen through Alibaba Cloud Model Studio
    OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured.")

        self.model = model or os.getenv(
            "QWEN_MODEL",
            "qwen3.8-max",
        )

        self.base_url = base_url or os.getenv(
            "QWEN_BASE_URL",
            ("https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

        print("[VehicleMind] LLM provider: Qwen")

        print(f"[VehicleMind] LLM model: {self.model}")

    def chat(
        self,
        messages,
        tools=None,
    ) -> LLMResponse:

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            # Function calling is easier to
            # maintain without preserved CoT.
            "extra_body": {
                "enable_thinking": False,
            },
        }

        if tools:
            kwargs["tools"] = tools

            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]

        message = choice.message

        normalized_calls = []

        for call in message.tool_calls or []:
            arguments_json = call.function.arguments or "{}"

            try:
                arguments = json.loads(arguments_json)

            except json.JSONDecodeError:
                arguments = {}

            normalized_calls.append(
                LLMToolCall(
                    id=call.id,
                    name=(call.function.name),
                    arguments=(arguments),
                    arguments_json=(arguments_json),
                )
            )

        return LLMResponse(
            content=(message.content),
            tool_calls=(normalized_calls),
            finish_reason=(choice.finish_reason),
            response_model=getattr(response, "model", None),
            usage=(
                {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
                if getattr(response, "usage", None) is not None
                else None
            ),
        )

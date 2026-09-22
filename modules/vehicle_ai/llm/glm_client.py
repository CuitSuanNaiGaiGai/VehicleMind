from __future__ import annotations

import json
import os

from zhipuai import ZhipuAI

from modules.vehicle_ai.llm.base import (
    BaseLLMClient,
    LLMResponse,
    LLMToolCall,
)


class GLMClient(BaseLLMClient):
    """
    Zhipu GLM client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.getenv("GLM_API_KEY")

        if not self.api_key:
            raise RuntimeError("GLM_API_KEY is not configured.")

        self.model = model or os.getenv(
            "GLM_MODEL",
            "glm-4.5",
        )

        self.base_url = base_url or os.getenv(
            "GLM_BASE_URL",
            ("https://open.bigmodel.cn/api/paas/v4"),
        )

        self.client = ZhipuAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        print("[VehicleMind] LLM provider: GLM")

        print(f"[VehicleMind] LLM model: {self.model}")

    def chat(
        self,
        messages,
        tools=None,
    ) -> LLMResponse:

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }

        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]

        message = choice.message

        normalized_calls = []

        for call in message.tool_calls or []:
            # Only process normal function calls.
            if (
                getattr(
                    call,
                    "function",
                    None,
                )
                is None
            ):
                continue

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
        )

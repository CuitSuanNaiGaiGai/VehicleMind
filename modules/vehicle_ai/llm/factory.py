from __future__ import annotations

import os

from modules.vehicle_ai.llm.base import (
    BaseLLMClient,
)

from modules.vehicle_ai.llm.glm_client import (
    GLMClient,
)

from modules.vehicle_ai.llm.qwen_client import (
    QwenClient,
)


def build_llm_client(
    provider: str | None = None,
    *,
    model: str | None = None,
    timeout_seconds: float = 30.0,
    temperature: float = 0.2,
) -> BaseLLMClient:

    selected_provider = provider or os.getenv(
        "VEHICLEMIND_LLM_PROVIDER",
        "qwen",
    )
    provider = (selected_provider or "qwen").strip().lower()

    if provider == "qwen":
        return QwenClient(
            model=model, timeout_seconds=timeout_seconds, temperature=temperature
        )

    if provider == "glm":
        return GLMClient(
            model=model, timeout_seconds=timeout_seconds, temperature=temperature
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

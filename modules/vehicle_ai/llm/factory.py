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
) -> BaseLLMClient:

    provider = provider or os.getenv(
        "VEHICLEMIND_LLM_PROVIDER",
        "qwen",
    )

    provider = provider.strip().lower()

    if provider == "qwen":
        return QwenClient()

    if provider == "glm":
        return GLMClient()

    raise ValueError(f"Unsupported LLM provider: {provider}")

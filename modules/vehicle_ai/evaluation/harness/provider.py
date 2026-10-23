"""promptfoo Python provider routing eval cases to VehicleMind backends."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from modules.vehicle_ai.evaluation.harness.bridge import (  # noqa: E402
    agent_eval_case,
)
from modules.vehicle_ai.evaluation.harness.injection import (  # noqa: E402
    DEFAULT_SECURITY_SUITE,
    injection_gate_case,
)


def _repo_relative(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def call_api(prompt: str, options: dict, context: dict) -> dict[str, Any]:
    """promptfoo provider entry; case identity arrives through vars.case_id."""
    config = options.get("config") or {}
    mode = str(config.get("mode", "agent_eval"))
    variables = context.get("vars") or {}
    case_id = str(variables.get("case_id", "")).strip()
    if not case_id:
        return {"error": "vars.case_id is required"}

    load_dotenv(_REPO_ROOT / ".env")
    try:
        if mode == "injection_gate":
            result = injection_gate_case(
                case_id,
                _repo_relative(
                    str(config.get("security_suite", DEFAULT_SECURITY_SUITE))
                ),
            )
        elif mode == "agent_eval":
            model = config.get("model")
            result = agent_eval_case(
                case_id,
                _repo_relative(
                    str(config.get("dataset_root", "scenarios/agent_eval/golden"))
                ),
                backend=str(config.get("backend", "qwen")),
                model=str(model) if model is not None else None,
            )
        else:
            return {"error": f"unsupported harness mode: {mode}"}
    except Exception as exc:  # a provider crash would abort the whole eval run
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"output": json.dumps(result, ensure_ascii=False)}

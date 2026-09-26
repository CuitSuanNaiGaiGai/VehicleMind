import json
from pathlib import Path

from modules.vehicle_ai.evaluation import batch_cli
from modules.vehicle_ai.evaluation.batch import load_frozen_cases
from modules.vehicle_ai.llm.base import BaseLLMClient, LLMResponse


ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "agent_eval" / "golden"


class ReplyClient(BaseLLMClient):
    def chat(self, messages, tools=None):
        return LLMResponse("当前观测可用。", [])


def test_batch_cli_runs_selected_case_without_online_credentials(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        batch_cli, "build_llm_client", lambda *args, **kwargs: ReplyClient()
    )
    assert (
        batch_cli.main(
            [
                "--provider",
                "qwen",
                "--model",
                "stub",
                "--golden",
                str(ROOT),
                "--output-root",
                str(tmp_path),
                "--case-id",
                "C01",
                "--split",
                "dev",
                "--repetitions",
                "1",
            ]
        )
        == 0
    )
    runs = list(tmp_path.glob("*/run.json"))
    assert len(runs) == 1
    assert len(load_frozen_cases(ROOT)) == 40
    provenance = json.loads(runs[0].read_text())["provenance"]
    assert provenance["case_count"] == 1
    assert provenance["repetitions"] == 1
    assert len(provenance["source_revision"]) == 40
    assert "api_key" not in provenance

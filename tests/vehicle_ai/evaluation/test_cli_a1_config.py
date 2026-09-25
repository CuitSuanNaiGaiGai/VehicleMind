from pathlib import Path
from types import SimpleNamespace

from modules.vehicle_ai.evaluation import cli
from modules.vehicle_ai.agent.budget import AgentBudgetConfig


def test_single_case_cli_uses_configured_rounds_when_flag_is_omitted(
    tmp_path, monkeypatch
):
    captured = {}

    class Client:
        model = "stub"

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "build_llm_client", lambda *args, **kwargs: Client())
    monkeypatch.setattr(
        cli,
        "run_trial",
        lambda *args, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setattr(cli, "grade_trial", lambda *args: {"status": "needs_review"})
    monkeypatch.setattr(cli, "write_report", lambda *args: None)

    case = Path("scenarios/agent_eval/candidates/C03.yaml")
    assert (
        cli.main(
            [
                str(case),
                "--provider",
                "qwen",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert captured["max_tool_rounds"] == AgentBudgetConfig.load().max_tool_rounds

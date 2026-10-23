from __future__ import annotations

import json

import pytest

from scripts.export_promptfoo_config import (
    REPO_ROOT,
    build_agent_eval_config,
    build_injection_config,
)


GOLDEN_ROOT = REPO_ROOT / "scenarios/agent_eval/golden"
CONFIG_DIR = REPO_ROOT / "eval/promptfoo"


def test_injection_config_covers_every_suite_case_in_order() -> None:
    config = build_injection_config()

    assert config["providers"][0]["config"]["mode"] == "injection_gate"
    ids = [test["vars"]["case_id"] for test in config["tests"]]
    assert ids == [
        "PI_USER_OVERRIDE",
        "PI_FAKE_CONFIRMATION",
        "PI_QUOTED_INSTRUCTION",
        "PI_RETRIEVED_EVIDENCE",
    ]
    assert config["defaultTest"]["assert"][0]["value"].endswith(
        ":assert_injection_gate"
    )
    for test in config["tests"]:
        expected = json.loads(test["vars"]["expected_json"])
        assert set(expected) == {
            "tool",
            "error",
            "navigation_state",
            "pending_action",
            "unconfirmed_sensitive_writes",
        }


def test_agent_config_covers_the_frozen_golden_set_deterministically() -> None:
    first = build_agent_eval_config(GOLDEN_ROOT)
    second = build_agent_eval_config(GOLDEN_ROOT)
    assert first == second
    assert len(first["tests"]) == 40
    assert {provider["label"] for provider in first["providers"]} == {"qwen", "glm"}
    splits = {test["description"].split(" · ")[-1] for test in first["tests"]}
    assert splits == {"dev", "heldout"}
    # The mechanical config stays minimal: no judge assertion, no dataset var.
    assert len(first["defaultTest"]["assert"]) == 1
    assert all("dataset_root" not in test["vars"] for test in first["tests"])


def test_rubric_config_adds_semantic_judging_deterministically() -> None:
    first = build_agent_eval_config(GOLDEN_ROOT, judged=True)
    second = build_agent_eval_config(GOLDEN_ROOT, judged=True)
    assert first == second
    assert len(first["tests"]) == 40
    assert len(first["defaultTest"]["assert"]) == 2
    assert first["defaultTest"]["assert"][1]["value"].endswith(
        ":assert_rubric_semantics"
    )
    assert first["defaultTest"]["assert"][1]["timeout"] == 600
    assert all(
        test["vars"]["dataset_root"] == "scenarios/agent_eval/golden"
        for test in first["tests"]
    )
    assert "rubric 语义判分" in first["description"]


def test_referenced_provider_and_assertion_paths_resolve() -> None:
    config = build_injection_config()
    provider_ref = config["providers"][0]["id"].removeprefix("file://")
    assert (CONFIG_DIR / provider_ref).resolve().is_file()
    assertion_ref = (
        config["defaultTest"]["assert"][0]["value"]
        .removeprefix("file://")
        .rsplit(":", 1)[0]
    )
    assert (CONFIG_DIR / assertion_ref).resolve().is_file()


def test_missing_dataset_root_is_rejected() -> None:
    with pytest.raises(Exception):
        build_agent_eval_config(REPO_ROOT / "scenarios/agent_eval/missing")

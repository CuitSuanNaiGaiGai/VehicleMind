from apps.vehicle_ai_demo.replay_demo import parse_args


def test_replay_cli_keeps_argument_names_and_has_chinese_help(capsys) -> None:
    args = parse_args(["--scenario", "assets/scenarios/drowsy_rest_stop.yaml"])
    assert args.scenario.name == "drowsy_rest_stop.yaml"

    try:
        parse_args(["--help"])
    except SystemExit as exit_error:
        assert exit_error.code == 0
    help_text = capsys.readouterr().out
    assert "离线回放" in help_text
    assert "--scenario" in help_text

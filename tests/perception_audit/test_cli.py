from __future__ import annotations

from pathlib import Path

from scripts import audit_perception_videos


def test_cli_accepts_explicit_input_directories(monkeypatch, capsys, tmp_path: Path):
    observed = {}

    def fake_run(cabin, road, output, **kwargs):
        observed.update(cabin=cabin, road=road, output=output, kwargs=kwargs)
        return tmp_path / "runs" / "finished"

    monkeypatch.setattr(audit_perception_videos, "run_audit", fake_run)
    exit_code = audit_perception_videos.main(
        [
            "--cabin-dir",
            str(tmp_path / "cabin"),
            "--road-dir",
            str(tmp_path / "road"),
            "--output-root",
            str(tmp_path / "runs"),
            "--sample-interval",
            "10",
        ]
    )
    assert exit_code == 0
    assert observed["cabin"] == tmp_path / "cabin"
    assert observed["road"] == tmp_path / "road"
    assert observed["kwargs"]["sample_interval"] == 10
    assert "report.html" in capsys.readouterr().out


def test_cli_explains_missing_model_without_traceback(monkeypatch, capsys, tmp_path):
    def fail(*_args, **_kwargs):
        raise FileNotFoundError("private/full/path/model.onnx")

    monkeypatch.setattr(audit_perception_videos, "run_audit", fail)
    exit_code = audit_perception_videos.main(
        [
            "--cabin-dir",
            str(tmp_path / "cabin"),
            "--road-dir",
            str(tmp_path / "road"),
        ]
    )
    assert exit_code == 2
    error = capsys.readouterr().err
    assert "模型或感知依赖" in error
    assert "private/full/path" not in error

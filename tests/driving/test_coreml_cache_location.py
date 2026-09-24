from pathlib import Path

from modules.driving.perception.runtime import coreml_cache_directory


def test_cpu_mode_never_creates_a_coreml_cache_directory(tmp_path: Path) -> None:
    model = tmp_path / "read_only_models" / "model.onnx"
    cache = coreml_cache_directory(
        model_path=model,
        prefer_coreml=False,
        available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        override=tmp_path / "results" / ".coreml_cache",
    )
    assert cache is None
    assert not model.parent.exists()


def test_coreml_cache_can_be_kept_outside_model_directory(tmp_path: Path) -> None:
    model = tmp_path / "models" / "model.onnx"
    override = tmp_path / "results" / ".coreml_cache"
    assert (
        coreml_cache_directory(
            model_path=model,
            prefer_coreml=True,
            available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            override=override,
        )
        == override
    )
    assert not model.parent.exists()

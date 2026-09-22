from pathlib import Path

import pytest

from apps.driving_demo.scene_summary import print_performance_summary


def test_zero_frame_run_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="No video frames were processed"):
        print_performance_summary(
            frame_count=0,
            source_fps=30.0,
            total_elapsed=0.1,
            output_path=Path("empty.mp4"),
            preprocess_ms=0.0,
            inference_ms=0.0,
            postprocess_ms=0.0,
            detector_ms=0.0,
            pipeline_ms=0.0,
        )

    assert "READY" not in capsys.readouterr().out

from __future__ import annotations

from pathlib import Path

import pytest

from apps.vehicle_ai_demo import integrated_demo
from apps.vehicle_ai_demo.integrated_demo import parse_args


def test_integrated_demo_accepts_offline_video_and_independent_rates() -> None:
    args = parse_args(
        [
            "--cabin-video",
            "cabin.mp4",
            "--road-video",
            "road.mp4",
            "--cabin-model",
            "face.task",
            "--road-model",
            "road.onnx",
            "--cabin-hz",
            "12",
            "--road-hz",
            "8",
            "--perception-only",
            "--duration",
            "3",
        ]
    )

    assert args.cabin_video == Path("cabin.mp4")
    assert args.road_video == Path("road.mp4")
    assert args.cabin_model == Path("face.task")
    assert args.road_model == Path("road.onnx")
    assert args.cabin_hz == 12
    assert args.road_hz == 8
    assert args.perception_only is True


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_integrated_demo_rejects_invalid_inference_rate(rate: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--cabin-hz", rate])


def test_perception_only_exits_nonzero_when_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedPipeline:
        def health(self) -> dict[str, str]:
            return {"last_error": "inference: RuntimeError: failed"}

        def stop(self) -> None:
            pass

        def join(self, timeout: float) -> None:
            pass

    pipeline = FailedPipeline()
    monkeypatch.setattr(
        integrated_demo, "_start_pipelines", lambda runtime, args: (pipeline, pipeline)
    )
    monkeypatch.setattr(integrated_demo.time, "sleep", lambda seconds: None)

    assert integrated_demo.main(["--perception-only", "--duration", "1"]) == 1

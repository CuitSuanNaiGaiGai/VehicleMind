import pytest

from modules.driving.benchmark.stats import summarize_ms, throughput_fps


def test_linear_percentiles_and_wall_clock_fps() -> None:
    summary = summarize_ms([10.0, 20.0, 30.0, 40.0])
    assert summary == {
        "count": 4,
        "mean": 25.0,
        "p50": 25.0,
        "p95": 38.5,
        "min": 10.0,
        "max": 40.0,
    }
    assert throughput_fps(30, 2.0) == 15.0


@pytest.mark.parametrize("values", [[], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_samples_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        summarize_ms(values)


@pytest.mark.parametrize(
    ("frames", "elapsed"),
    [(0, 1.0), (-1, 1.0), (1, 0.0), (1, -1.0), (1, float("nan"))],
)
def test_invalid_throughput_inputs_are_rejected(frames: int, elapsed: float) -> None:
    with pytest.raises(ValueError):
        throughput_fps(frames, elapsed)

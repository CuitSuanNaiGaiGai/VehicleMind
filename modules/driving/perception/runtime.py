from __future__ import annotations

import time

import numpy as np


def warm_up_session(
    session,
    *,
    input_name: str,
    input_height: int,
    input_width: int,
    runs: int,
) -> None:
    """Warm up ONNX/CoreML execution before latency measurements."""
    print(f"[VehicleMind] Warming up YOLOPv2 ({runs} runs)...")
    dummy = np.zeros(
        (1, 3, input_height, input_width),
        dtype=np.float32,
    )
    for index in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"[VehicleMind] Warmup {index + 1}: {elapsed_ms:.1f} ms")

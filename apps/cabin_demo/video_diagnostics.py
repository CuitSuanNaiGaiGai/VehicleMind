from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def print_cabin_diagnostics(
    *,
    output_video_path: Path,
    eye_analyzer: Any,
    diag_face_frames: int,
    frame_index: int,
    diag_ear_values: list[float],
    diag_closed_frames: int,
    diag_max_closure_ms: int,
    diag_max_perclos: float,
    diag_max_ready_perclos: float,
    diag_suspected_frames: int,
    diag_drowsy_frames: int,
    first_drowsy_event_time: float | None,
) -> None:
    # ========================================================
    # Result MP4
    # ========================================================

    print()

    print(f"[VehicleMind] Processed video saved: {output_video_path}")

    # ========================================================
    # Diagnostics
    # ========================================================

    print()
    print("=" * 60)
    print("VehicleMind Cabin Diagnostics")
    print("=" * 60)

    print(
        f"Face detection coverage: "
        f"{diag_face_frames}/{frame_index} "
        f"("
        f"{diag_face_frames / max(frame_index, 1) * 100:.1f}%"
        f")"
    )

    if diag_ear_values:
        ear_array = np.asarray(
            diag_ear_values,
            dtype=np.float32,
        )

        percentiles = np.percentile(
            ear_array,
            [
                1,
                5,
                10,
                25,
                50,
                75,
                90,
                95,
                99,
            ],
        )

        print()
        print("EAR statistics")
        print("-" * 40)

        print(f"Configured EAR threshold: {eye_analyzer.ear_threshold:.3f}")

        print(f"EAR min:    {ear_array.min():.3f}")

        print(f"EAR mean:   {ear_array.mean():.3f}")

        print(f"EAR median: {np.median(ear_array):.3f}")

        labels = [
            "P01",
            "P05",
            "P10",
            "P25",
            "P50",
            "P75",
            "P90",
            "P95",
            "P99",
        ]

        for label, value in zip(
            labels,
            percentiles,
        ):
            print(f"{label}: {value:.3f}")

        print()

        print(f"Frames classified CLOSED: {diag_closed_frames}")

        print(
            "Closed-frame ratio: "
            f"{diag_closed_frames / max(diag_face_frames, 1) * 100:.1f}%"
        )

    else:
        print()
        print("No valid EAR observations.")

    print()
    print("Temporal statistics")
    print("-" * 40)

    print(f"Maximum continuous closure: {diag_max_closure_ms / 1000:.2f}s")

    print(f"Maximum PERCLOS (all): {diag_max_perclos * 100:.1f}%")

    print(f"Maximum PERCLOS (ready): {diag_max_ready_perclos * 100:.1f}%")

    print(f"SUSPECTED frames: {diag_suspected_frames}")

    print(f"DROWSY frames: {diag_drowsy_frames}")

    if first_drowsy_event_time is not None:
        print(f"First DROWSY event: {first_drowsy_event_time:.2f}s")

    print("=" * 60)

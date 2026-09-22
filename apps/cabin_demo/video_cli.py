import argparse
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"


# ============================================================
# Arguments
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(description="VehicleMind Cabin Intelligence Demo")

    parser.add_argument(
        "--video",
        type=str,
        default="assets/demo/MicroSleep_test.mp4",
        help="Input public driver video",
    )

    parser.add_argument(
        "--output-video",
        type=str,
        default="assets/demo/MicroSleep_result.mp4",
        help="Processed result video",
    )

    parser.add_argument(
        "--output-gif",
        type=str,
        default="assets/demo/cabin_demo.gif",
        help="GitHub README GIF",
    )

    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="GIF frame rate",
    )

    parser.add_argument(
        "--gif-width",
        type=int,
        default=960,
        help="GIF width",
    )

    parser.add_argument(
        "--pre-event",
        type=float,
        default=3.0,
        help="Seconds retained before first DROWSY event",
    )

    parser.add_argument(
        "--post-event",
        type=float,
        default=5.0,
        help="Seconds retained after first DROWSY event",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization while processing",
    )

    return parser.parse_args()

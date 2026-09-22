import shutil
import subprocess
from pathlib import Path


def check_ffmpeg() -> str:
    """
    Locate FFmpeg executable.
    """

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg was not found.\nInstall it first with:\n    brew install ffmpeg"
        )

    return ffmpeg_path


def generate_gif_with_ffmpeg(
    input_video: Path,
    output_gif: Path,
    event_time: float,
    pre_event: float,
    post_event: float,
    gif_fps: int,
    gif_width: int,
) -> None:
    """
    Generate a high-quality GIF from the processed MP4.

    FFmpeg creates one optimized global palette using
    palettegen and applies it using paletteuse.

    Compared with Pillow per-frame quantization, this
    substantially improves:
        - skin tone
        - dark cabin regions
        - green outdoor areas
        - temporal color consistency
    """

    ffmpeg = check_ffmpeg()

    output_gif.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = max(
        0.0,
        event_time - pre_event,
    )

    duration = pre_event + post_event

    print()
    print("[VehicleMind] Generating GitHub GIF...")

    print(
        f"[VehicleMind] GIF window: {start_time:.2f}s -> {start_time + duration:.2f}s"
    )

    print(f"[VehicleMind] GIF FPS: {gif_fps}")

    print(f"[VehicleMind] GIF width: {gif_width}px")

    # --------------------------------------------------------
    # High-quality GIF filter
    #
    # Processing:
    #
    # MP4
    #  ↓
    # fps
    #  ↓
    # Lanczos resize
    #  ↓
    # split
    #  ├─ palettegen
    #  └─ original frames
    #         ↓
    #     paletteuse
    #         ↓
    #        GIF
    #
    # One global palette is generated for the selected
    # event window.
    # --------------------------------------------------------

    filter_complex = (
        f"[0:v]"
        f"fps={gif_fps},"
        f"scale={gif_width}:-1:"
        f"flags=lanczos,"
        f"split[s0][s1];"
        f"[s0]"
        f"palettegen="
        f"max_colors=256:"
        f"stats_mode=diff[p];"
        f"[s1][p]"
        f"paletteuse="
        f"dither=sierra2_4a:"
        f"diff_mode=rectangle"
    )

    command = [
        ffmpeg,
        "-y",
        # Input result video
        "-i",
        str(input_video),
        # Accurate event-centered seek
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        filter_complex,
        "-loop",
        "0",
        str(output_gif),
    ]

    try:
        subprocess.run(
            command,
            check=True,
        )

    except subprocess.CalledProcessError as exc:
        raise RuntimeError("FFmpeg failed while generating GIF.") from exc

    if not output_gif.exists():
        raise RuntimeError("FFmpeg finished but GIF file was not created.")

    gif_size_mb = output_gif.stat().st_size / (1024 * 1024)

    print(f"[VehicleMind] High-quality GIF saved: {output_gif}")

    print(f"[VehicleMind] GIF size: {gif_size_mb:.2f} MB")

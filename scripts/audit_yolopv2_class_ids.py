"""Record raw YOLOPv2 head shapes and NMS class IDs without assigning labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.driving.perception import yolopv2_utils
from modules.driving.perception.preprocess import preprocess_frame

CONFIDENCE_THRESHOLD = 0.30
NMS_IOU_THRESHOLD = 0.45
DEFAULT_DYNAMIC_INPUT_SIZE = 640


def sample_frame_indices(frame_count: int) -> list[int]:
    """Return unique first, middle, and last indices for a non-empty video."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    return sorted({0, frame_count // 2, frame_count - 1})


def count_class_ids(detections: Sequence[Sequence[float]]) -> dict[str, int]:
    """Count NMS class IDs while deliberately preserving no semantic names."""
    counts = Counter(str(int(row[5])) for row in detections)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def safe_asset_reference(path: Path, sha256: str) -> dict[str, str]:
    """Expose a stable asset identity without serializing its local path."""
    return {"filename": path.name, "sha256": sha256}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as asset:
        for chunk in iter(lambda: asset.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sampled_frames(video_path: Path) -> tuple[int, dict[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("video could not be opened")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(frame_count)
        frames: dict[int, np.ndarray] = {}
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError("sample frame could not be decoded")
            frames[index] = frame
    finally:
        capture.release()
    return frame_count, frames


def _input_size(input_shape: Sequence[Any]) -> tuple[int, int]:
    if len(input_shape) >= 4:
        height, width = input_shape[2], input_shape[3]
        if (
            isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
            and isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
        ):
            return height, width
    return DEFAULT_DYNAMIC_INPUT_SIZE, DEFAULT_DYNAMIC_INPUT_SIZE


def _head_arrays(raw_output: Any) -> list[np.ndarray]:
    if isinstance(raw_output, (list, tuple)):
        heads = list(raw_output)
    elif isinstance(raw_output, np.ndarray) and raw_output.shape[0] == 3:
        heads = [raw_output[index] for index in range(3)]
    else:
        raise ValueError("model did not return three detection heads")
    if len(heads) != 3:
        raise ValueError("model did not return three detection heads")
    return [np.asarray(head) for head in heads]


def _inspect_model(
    model_path: Path,
    frames: dict[int, np.ndarray],
    *,
    onnxruntime: Any,
) -> dict[str, Any]:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    input_height, input_width = _input_size(input_meta.shape)
    output_samples: list[dict[str, Any]] = []
    observed_head_shapes: list[list[int]] | None = None
    decoded_shape: list[int] | None = None

    for frame_index, frame in frames.items():
        tensor, _, _ = preprocess_frame(
            frame,
            input_height=input_height,
            input_width=input_width,
            static_input=True,
        )
        outputs = session.run(None, {input_meta.name: tensor})
        heads = _head_arrays(outputs[0])
        if len(outputs) < 4:
            raise ValueError("model output is missing anchor tensors")
        prediction = yolopv2_utils.split_for_trace_model(
            [head.copy() for head in heads],
            [outputs[index] for index in range(1, 4)],
        )
        detections = yolopv2_utils.non_max_suppression(
            prediction,
            conf_thres=CONFIDENCE_THRESHOLD,
            iou_thres=NMS_IOU_THRESHOLD,
        )[0]
        observed_head_shapes = [list(head.shape) for head in heads]
        decoded_shape = list(prediction.shape)
        output_samples.append(
            {
                "frame_index": frame_index,
                "detection_count": len(detections),
                "class_id_counts": count_class_ids(detections),
            }
        )

    return {
        "asset": safe_asset_reference(model_path, _sha256(model_path)),
        "input_name": input_meta.name,
        "input_shape": list(input_meta.shape),
        "inference_size": [input_height, input_width],
        "providers": session.get_providers(),
        "raw_head_shapes": observed_head_shapes,
        "decoded_shape": decoded_shape,
        "samples": output_samples,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="记录 YOLOPv2 原始 head shape 与 NMS class ID，不推断类别语义"
    )
    parser.add_argument("--video", type=Path, required=True, help="待抽帧视频")
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个本地 YOLOPv2 ONNX 文件",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import onnxruntime

        frame_count, frames = _load_sampled_frames(args.video)
        report = {
            "schema_version": 1,
            "video": {
                **safe_asset_reference(args.video, _sha256(args.video)),
                "frame_count": frame_count,
                "sampled_frame_indices": list(frames),
            },
            "runtime": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "onnxruntime": onnxruntime.__version__,
            },
            "thresholds": {
                "confidence": CONFIDENCE_THRESHOLD,
                "nms_iou": NMS_IOU_THRESHOLD,
            },
            "models": [
                _inspect_model(model, frames, onnxruntime=onnxruntime)
                for model in args.models
            ],
            "semantic_mapping": "not_verified",
            "accuracy": "not_evaluated_without_ground_truth_labels",
        }
    except Exception as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "detail": "audit did not complete"},
                ensure_ascii=False,
            )
        )
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

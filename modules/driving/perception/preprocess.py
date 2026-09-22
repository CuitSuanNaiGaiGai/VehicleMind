from __future__ import annotations

import numpy as np

from modules.driving.perception import yolopv2_utils


def preprocess_frame(
    frame: np.ndarray,
    *,
    input_height: int,
    input_width: int,
    static_input: bool,
) -> tuple[np.ndarray, object, tuple[float, float]]:
    """Convert an OpenCV BGR frame to a YOLOPv2 NCHW tensor."""
    image, ratio, (pad_w, pad_h) = yolopv2_utils.letterbox(
        frame.copy(),
        new_shape=(input_height, input_width),
        auto=not static_input,
        scaleFill=False,
        scaleup=True,
        stride=32,
    )
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32, copy=False) / 255.0
    image = np.expand_dims(image, axis=0)
    return image, ratio, (pad_w, pad_h)

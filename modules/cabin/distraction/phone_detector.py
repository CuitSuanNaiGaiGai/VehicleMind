from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PhoneDetection:
    """
    One detected cell phone.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    confidence: float

    class_id: int
    class_name: str


@dataclass
class PhoneDetectionResult:
    """
    Cell-phone detections for one frame.
    """

    detected: bool

    detections: List[PhoneDetection]

    best_detection: Optional[PhoneDetection]


class PhoneDetector:
    """
    Detect cell phones using a COCO-pretrained YOLO model.

    Important:
        This module only answers:

            "Is a cell phone visible in this frame?"

        It does NOT directly classify driver distraction.

    PHONE_USE / DISTRACTED will be handled later by
    spatial association + temporal behavior reasoning.
    """

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        confidence_threshold: float = 0.35,
        image_size: int = 640,
        device: Optional[str] = None,
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.image_size = image_size

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

        if device is None:

            if (
                torch.backends.mps.is_available()
            ):
                self.device = "mps"

            elif torch.cuda.is_available():
                self.device = "cuda"

            else:
                self.device = "cpu"

        else:
            self.device = device

        print(
            "[VehicleMind] "
            f"PhoneDetector device: {self.device}"
        )

        # -----------------------------------------------------
        # YOLO
        # -----------------------------------------------------

        self.model = YOLO(
            model_name
        )

        # -----------------------------------------------------
        # Do NOT hard-code the COCO class index.
        #
        # Resolve "cell phone" from model metadata so that
        # the detector remains less dependent on a specific
        # class-index ordering.
        # -----------------------------------------------------

        self.phone_class_ids = []

        for class_id, name in (
            self.model.names.items()
        ):

            if (
                name.strip().lower()
                == "cell phone"
            ):

                self.phone_class_ids.append(
                    int(class_id)
                )

        if not self.phone_class_ids:

            raise RuntimeError(
                "The loaded YOLO model does not "
                "contain a 'cell phone' class."
            )

        print(
            "[VehicleMind] "
            "Cell-phone class IDs: "
            f"{self.phone_class_ids}"
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> PhoneDetectionResult:
        """
        Detect cell phones in one BGR OpenCV frame.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            classes=self.phone_class_ids,
            device=self.device,
            verbose=False,
        )

        detections: List[
            PhoneDetection
        ] = []

        if not results:

            return PhoneDetectionResult(
                detected=False,
                detections=[],
                best_detection=None,
            )

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            return PhoneDetectionResult(
                detected=False,
                detections=[],
                best_detection=None,
            )

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            class_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )

            class_name = (
                self.model.names[
                    class_id
                ]
            )

            detection = PhoneDetection(
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )

            detections.append(
                detection
            )

        best_detection = max(
            detections,
            key=lambda item: (
                item.confidence
            ),
        )

        return PhoneDetectionResult(
            detected=True,
            detections=detections,
            best_detection=best_detection,
        )

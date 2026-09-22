# Third-party notices

VehicleMind's original code is licensed under Apache License 2.0. The project does not commit the model files listed below; users obtain them separately and remain responsible for the upstream terms. The VehicleMind license does not replace or relicense third-party software, models, datasets, or media.

## MediaPipe Face Landmarker

- Purpose: face landmark inference for cabin perception.
- Software project: https://github.com/google-ai-edge/mediapipe
- Model guide: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
- Pinned artifact and hash: see `assets/model_manifest.yaml`.
- Repository license: Apache License 2.0.
- Review note: the exact task-bundle page does not currently provide artifact-specific redistribution language. The repository therefore records the license conservatively as `review-required-for-redistribution` and does not redistribute the bundle.

## YOLOPv2

- Purpose: joint object, lane, and drivable-area perception.
- Upstream: https://github.com/CAIC-AD/YOLOPv2
- Release: https://github.com/CAIC-AD/YOLOPv2/releases/tag/V0.0.1
- License: MIT, https://github.com/CAIC-AD/YOLOPv2/blob/main/LICENSE
- Review note: the three pinned ONNX files are legacy local conversions. Their bytes are pinned, but the exact export command for the dynamic ONNX file still needs to be reconstructed before a resume-grade release claim.

## Ultralytics YOLO26

- Purpose: optional phone detection demo.
- Model: `yolo26n.pt`, loaded by the `ultralytics` package.
- Documentation: https://docs.ultralytics.com/models/yolo26
- License guidance: https://www.ultralytics.com/license
- License: AGPL-3.0 by default, or a separate Ultralytics Enterprise license.
- Review note: this dependency can materially affect the license obligations of the larger application. It remains optional and unresolved in the asset manifest; do not describe the project as permissively licensed while this path is distributed without a completed license review.

## Python packages

Runtime and development dependencies retain their own licenses. Exact resolved versions are recorded in `uv.lock`; review those licenses before distribution or commercial use.

from __future__ import annotations

from ultralytics import YOLO


def load_model(weights_path, device):
    """Load an Ultralytics detection model (e.g. patched YOLOv7 `.onnx`)."""
    _ = device  # applied in detect_frame via model.predict(device=...)
    return YOLO(str(weights_path), task="detect")

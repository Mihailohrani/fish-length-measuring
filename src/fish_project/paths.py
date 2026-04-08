from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = LOCAL_DATA_DIR / "models"
VIDEOS_DIR = LOCAL_DATA_DIR / "videos"
BAGS_DIR = LOCAL_DATA_DIR / "bags"
IMAGES_DIR = LOCAL_DATA_DIR / "images"
ORIGINAL_IMAGES_DIR = IMAGES_DIR / "original"
DOWNSAMPLED_IMAGES_DIR = IMAGES_DIR / "downsampled"
UPLOADS_DIR = LOCAL_DATA_DIR / "uploads"

YOLOV7_ONNX_PATH = MODELS_DIR / "FishYolov7_tiny_ultralytics.onnx"
YOLO26_FISH_PT_PATH = MODELS_DIR / "fish-yolo26n.pt"

DEFAULT_WEIGHTS_PATH = YOLOV7_ONNX_PATH


def detector_weight_choices() -> dict[str, str]:
    """Human-readable label → absolute path for weights that exist under ``data/models``."""
    out: dict[str, str] = {}
    if YOLOV7_ONNX_PATH.is_file():
        out["YOLOv7"] = str(YOLOV7_ONNX_PATH.resolve())
    if YOLO26_FISH_PT_PATH.is_file():
        out["YOLO26"] = str(YOLO26_FISH_PT_PATH.resolve())
    return out


def ensure_local_data_dirs() -> None:
    """Create the expected local working directories if they do not exist."""
    for path in (
        LOCAL_DATA_DIR,
        MODELS_DIR,
        VIDEOS_DIR,
        BAGS_DIR,
        IMAGES_DIR,
        ORIGINAL_IMAGES_DIR,
        DOWNSAMPLED_IMAGES_DIR,
        UPLOADS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_default_image_browser_dir() -> Path:
    """Return the preferred root directory for notebook image browsing."""
    if ORIGINAL_IMAGES_DIR.is_dir():
        return ORIGINAL_IMAGES_DIR
    if DOWNSAMPLED_IMAGES_DIR.is_dir():
        return DOWNSAMPLED_IMAGES_DIR
    return IMAGES_DIR

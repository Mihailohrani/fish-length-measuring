from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = LOCAL_DATA_DIR / "models"
VIDEOS_DIR = LOCAL_DATA_DIR / "videos"
IMAGES_DIR = LOCAL_DATA_DIR / "images"
UPLOADS_DIR = LOCAL_DATA_DIR / "uploads"

DEFAULT_WEIGHTS_PATH = MODELS_DIR / "FishYolov7_tiny_ultralytics.onnx"


def ensure_local_data_dirs() -> None:
    """Create the expected local working directories if they do not exist."""
    for path in (LOCAL_DATA_DIR, MODELS_DIR, VIDEOS_DIR, IMAGES_DIR, UPLOADS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_default_image_browser_dir() -> Path:
    """Return the preferred root directory for notebook image browsing."""
    for candidate in ("original", "downsampled"):
        candidate_path = IMAGES_DIR / candidate
        if candidate_path.is_dir():
            return candidate_path
    return IMAGES_DIR

"""Shared fish detection package."""

from .detection import detect_frame
from .measurement import measure_fish_contour
from .model import load_model
from .visualization import draw_detections

__all__ = [
    "detect_frame",
    "draw_detections",
    "load_model",
    "measure_fish_contour",
]

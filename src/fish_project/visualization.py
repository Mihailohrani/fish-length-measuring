from __future__ import annotations

import cv2
import numpy as np


def _format_dimension(value_px, px_per_unit, unit_name):
    if px_per_unit is None or px_per_unit <= 0:
        return f"{value_px:.0f}px"

    precision = 0 if unit_name == "px" else 1
    value = value_px / px_per_unit
    return f"{value:.{precision}f}{unit_name}"


def draw_detections(frame, detections, *, px_per_unit=None, unit_name="px"):
    """Draw bounding boxes, rotated rectangles, and size annotations on a frame."""
    h = frame.shape[0]
    scale = max(h / 1200, 1.0)
    line_th = max(round(2 * scale), 1)
    font_scale = 0.5 * scale
    font_th = max(round(2 * scale), 1)
    pad = max(round(8 * scale), 4)

    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox"]
        conf = detection["confidence"]
        rotated_rect = detection["rotated_rect"]

        length = _format_dimension(detection["length_px"], px_per_unit, unit_name)
        width = _format_dimension(detection["width_px"], px_per_unit, unit_name)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), line_th)

        if rotated_rect is not None:
            box_points = cv2.boxPoints(rotated_rect).astype(np.int32)
            cv2.drawContours(frame, [box_points], 0, (255, 255, 0), line_th)

        method = "" if detection["contour_found"] else " (bbox)"
        label = f"Fish {conf:.2f} | L:{length} W:{width}{method}"
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_th,
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - pad),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - pad // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_th,
        )

    return frame

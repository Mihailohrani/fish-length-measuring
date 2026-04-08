from __future__ import annotations

import cv2


def measure_fish_contour(
    frame,
    bbox,
    *,
    padding=5,
    canny_low=50,
    canny_high=150,
    min_contour_ratio=0.10,
):
    """Measure fish dimensions using contour analysis within a YOLO bounding box."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w_frame, x2 + padding)
    y2p = min(h_frame, y2 + padding)
    crop = frame[y1p:y2p, x1p:x2p]

    bw, bh = x2 - x1, y2 - y1
    fallback = {
        "length_px": float(max(bw, bh)),
        "width_px": float(min(bw, bh)),
        "angle": 0.0,
        "contour_found": False,
        "rotated_rect": None,
        "edges": None,
    }

    if crop.size == 0:
        return fallback

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        fallback["edges"] = edges
        return fallback

    largest = max(contours, key=cv2.contourArea)
    crop_area = (x2p - x1p) * (y2p - y1p)
    if cv2.contourArea(largest) < min_contour_ratio * crop_area:
        fallback["edges"] = edges
        return fallback

    rect = cv2.minAreaRect(largest)
    (cx_local, cy_local), (rect_w, rect_h), angle = rect
    length_px = max(rect_w, rect_h)
    width_px = min(rect_w, rect_h)

    cx_frame = cx_local + x1p
    cy_frame = cy_local + y1p
    rotated_rect_frame = ((cx_frame, cy_frame), (rect_w, rect_h), angle)

    return {
        "length_px": float(length_px),
        "width_px": float(width_px),
        "angle": float(angle),
        "contour_found": True,
        "rotated_rect": rotated_rect_frame,
        "edges": edges,
    }

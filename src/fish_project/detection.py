from __future__ import annotations

import numpy as np
import torch

from .measurement import measure_fish_contour


def _preprocess_frame(
    frame,
    *,
    use_clahe=False,
    clahe_clip=2.0,
    use_bilateral=False,
    bilateral_sigma=50,
):
    import cv2

    processed = frame.copy()

    if use_bilateral:
        processed = cv2.bilateralFilter(
            processed,
            d=9,
            sigmaColor=bilateral_sigma,
            sigmaSpace=bilateral_sigma,
        )

    if use_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        processed = cv2.cvtColor(
            cv2.merge([l_channel, a_channel, b_channel]),
            cv2.COLOR_LAB2BGR,
        )

    return processed


def _predict_device(device: torch.device) -> str | int:
    if device.type == "cpu":
        return "cpu"
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return str(device)


def detect_frame(
    model,
    frame,
    device,
    *,
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
    padding=5,
    canny_low=50,
    canny_high=150,
    min_contour_ratio=0.10,
    use_clahe=False,
    clahe_clip=2.0,
    use_bilateral=False,
    bilateral_sigma=50,
):
    """Run fish detection on a single frame and return a list of detections."""
    if frame is None or frame.size == 0:
        return []

    processed_frame = _preprocess_frame(
        frame,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        use_bilateral=use_bilateral,
        bilateral_sigma=bilateral_sigma,
    )

    results = model.predict(
        source=processed_frame,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=img_size,
        device=_predict_device(device),
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    else:
        boxes = np.asarray(boxes)

    confs = result.boxes.conf
    if hasattr(confs, "cpu"):
        confs = confs.cpu().numpy()
    else:
        confs = np.asarray(confs)

    clss = result.boxes.cls
    if hasattr(clss, "cpu"):
        clss = clss.cpu().numpy().astype(int)
    else:
        clss = np.asarray(clss, dtype=int)

    detections = []
    for row in range(boxes.shape[0]):
        x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[row]]
        conf = float(confs[row])
        cls_id = int(clss[row])
        bbox = (x1, y1, x2, y2)
        measurement = measure_fish_contour(
            frame,
            bbox,
            padding=padding,
            canny_low=canny_low,
            canny_high=canny_high,
            min_contour_ratio=min_contour_ratio,
        )
        detections.append(
            {
                "bbox": bbox,
                "confidence": conf,
                "class": cls_id,
                "length_px": measurement["length_px"],
                "width_px": measurement["width_px"],
                "angle": measurement["angle"],
                "contour_found": measurement["contour_found"],
                "rotated_rect": measurement["rotated_rect"],
                "edges": measurement["edges"],
            }
        )

    return detections

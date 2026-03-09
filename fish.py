import sys
import os
import glob

# Add yolov7 repo to path so torch.load() can unpickle the model classes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov7"))

import cv2
import torch
from torch import nn
import numpy as np
from models.common import Conv
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

def load_model(weights_path, device):
    """Load the YOLOv7 model from the original repo format checkpoint."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()

    # Compatibility fixes for newer PyTorch versions
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()

    if device.type != "cpu":
        model.half()
    return model

def measure_fish_contour(frame, bbox, padding=5):
    """Measure fish dimensions using contour analysis within a YOLO bounding box.

    Crops the bbox region, finds the largest contour, fits a minimum-area
    rotated rectangle, and returns precise length and width in pixels.
    Falls back to axis-aligned bbox dimensions if no valid contour is found.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Pad the crop to avoid clipping fish edges
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w_frame, x2 + padding)
    y2p = min(h_frame, y2 + padding)
    crop = frame[y1p:y2p, x1p:x2p]

    # Fallback values from axis-aligned bbox
    bw, bh = x2 - x1, y2 - y1
    fallback = {
        "length_px":     float(max(bw, bh)),
        "width_px":      float(min(bw, bh)),
        "angle":         0.0,
        "contour_found": False,
        "rotated_rect":  None,
    }

    if crop.size == 0:
        return fallback

    # Preprocess: grayscale → blur → edge detection → dilate
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    # Find contours, select the largest by area
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback

    largest = max(contours, key=cv2.contourArea)

    # Reject if contour is too small relative to the crop (likely noise)
    crop_area = (x2p - x1p) * (y2p - y1p)
    if cv2.contourArea(largest) < 0.10 * crop_area:
        return fallback

    # Fit minimum-area rotated rectangle
    rect = cv2.minAreaRect(largest)
    (cx_local, cy_local), (rect_w, rect_h), angle = rect

    length_px = max(rect_w, rect_h)
    width_px = min(rect_w, rect_h)

    # Convert center back to frame coordinates for drawing
    cx_frame = cx_local + x1p
    cy_frame = cy_local + y1p
    rotated_rect_frame = ((cx_frame, cy_frame), (rect_w, rect_h), angle)

    return {
        "length_px":     float(length_px),
        "width_px":      float(width_px),
        "angle":         float(angle),
        "contour_found": True,
        "rotated_rect":  rotated_rect_frame,
    }

def detect_frame(model, frame, device, img_size=640, conf_thres=0.25, iou_thres=0.45):
    """Run fish detection on a single frame. Returns list of detections."""
    # Preprocess: letterbox resize, BGR→RGB, HWC→CHW, normalize
    img = letterbox(frame, img_size, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if device.type != "cpu" else img.float()
    img /= 255.0
    img = img.unsqueeze(0)

    # Inference + NMS
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    detections = []
    for det in pred:
        if len(det):
            # Scale boxes back to original frame dimensions
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                bbox = (x1, y1, x2, y2)
                measurement = measure_fish_contour(frame, bbox)
                detections.append({
                    "bbox":          bbox,
                    "confidence":    float(conf),
                    "class":         int(cls),
                    "length_px":     measurement["length_px"],
                    "width_px":      measurement["width_px"],
                    "angle":         measurement["angle"],
                    "contour_found": measurement["contour_found"],
                    "rotated_rect":  measurement["rotated_rect"],
                })

    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes, rotated rectangles, and size annotations on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        length_px = det["length_px"]
        width_px = det["width_px"]
        rotated_rect = det["rotated_rect"]

        # Axis-aligned YOLO bounding box (green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Rotated rectangle from contour analysis (cyan)
        if rotated_rect is not None:
            box_points = cv2.boxPoints(rotated_rect).astype(int)
            cv2.drawContours(frame, [box_points], 0, (255, 255, 0), 2)

        # Label with both dimensions
        if det["contour_found"]:
            label = f"Fish {conf:.2f} | L:{length_px:.0f}px W:{width_px:.0f}px"
        else:
            label = f"Fish {conf:.2f} | L:{length_px:.0f}px W:{width_px:.0f}px (bbox)"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame

def process_videos(data_dir="data", weights_path="data/FishYolov7_tiny.pt", img_size=640):
    """Process all MP4 videos in the data directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {weights_path}...")
    model = load_model(weights_path, device)
    print("Model loaded successfully.")

    video_files = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
    if not video_files:
        print(f"No .mp4 files found in {data_dir}/")
        return

    print(f"Found {len(video_files)} video(s).\n")

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        print(f"Processing: {video_name}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  Error: Could not open {video_path}")
            continue

        frame_count = 0
        total_detections = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            detections = detect_frame(model, frame, device, img_size)
            total_detections += len(detections)

            if detections:
                print(f"  Frame {frame_count}: {len(detections)} fish detected")
                for i, det in enumerate(detections):
                    method = "contour" if det["contour_found"] else "bbox"
                    print(f"    Fish {i + 1}: conf={det['confidence']:.2f}, "
                          f"L={det['length_px']:.0f}px, W={det['width_px']:.0f}px, "
                          f"angle={det['angle']:.1f}°, method={method}")

            annotated = draw_detections(frame, detections)
            cv2.imshow(f"Fish Detection - {video_name}", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("\nPlayback stopped by user.")
                return

        cap.release()
        cv2.destroyAllWindows()
        print(f"  Finished: {frame_count} frames, {total_detections} total detections.\n")

    print("All videos processed.")

if __name__ == "__main__":
    process_videos()

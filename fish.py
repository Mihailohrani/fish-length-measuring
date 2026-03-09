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
                w = x2 - x1
                h = y2 - y1
                length_px = max(w, h)
                detections.append({
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "class":      int(cls),
                    "length_px":  length_px,
                })

    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and length annotations on the frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        length_px = det["length_px"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Fish {conf:.2f} | {length_px}px"
        # Background for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
                    print(f"    Fish {i + 1}: confidence={det['confidence']:.2f}, "
                          f"length={det['length_px']}px, bbox={det['bbox']}")

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

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Fish Detection & Measurement

    Interactive exploration of YOLOv7 fish detection with contour-based size measurement.
    """)
    return


@app.cell
def _():
    import sys
    import os
    import glob

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "yolov7"))
    # Also handle running from the project root
    if os.path.isdir("yolov7"):
        sys.path.insert(0, os.path.abspath("yolov7"))

    import cv2
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from torch import nn
    from models.common import Conv
    from utils.general import non_max_suppression, scale_coords
    from utils.datasets import letterbox

    return (
        Conv,
        cv2,
        glob,
        letterbox,
        nn,
        non_max_suppression,
        np,
        os,
        pd,
        plt,
        scale_coords,
        torch,
    )


@app.cell
def _(Conv, nn, torch):
    def load_model(weights_path, device):
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model = ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("data/FishYolov7_tiny.pt", device)
    print(f"Model loaded on {device}")
    return device, model


@app.cell
def _(glob, mo, os):
    video_files = sorted(glob.glob(os.path.join("data", "*.mp4")))
    video_dropdown = mo.ui.dropdown(
        options={os.path.basename(v): v for v in video_files},
        label="Video",
    )
    video_dropdown
    return (video_dropdown,)


@app.cell
def _(cv2, mo, video_dropdown):
    mo.stop(video_dropdown.value is None, mo.md("*Select a video above.*"))

    _cap = cv2.VideoCapture(video_dropdown.value)
    total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = _cap.get(cv2.CAP_PROP_FPS)
    _cap.release()

    frame_slider = mo.ui.slider(
        start=0,
        stop=max(0, total_frames - 1),
        step=1,
        value=0,
        label=f"Frame (0–{total_frames - 1})",
        full_width=True,
    )
    mo.md(f"**{video_dropdown.value}** — {total_frames} frames, {fps:.1f} FPS")
    return frame_slider, total_frames


@app.cell
def _(frame_slider):
    frame_slider
    return


@app.cell
def _(mo):
    conf_slider = mo.ui.slider(start=0.05, stop=0.95, step=0.05, value=0.25, label="Confidence threshold")
    iou_slider = mo.ui.slider(start=0.1, stop=0.9, step=0.05, value=0.45, label="IoU threshold")
    canny_low_slider = mo.ui.slider(start=10, stop=200, step=10, value=50, label="Canny low")
    canny_high_slider = mo.ui.slider(start=50, stop=300, step=10, value=150, label="Canny high")
    contour_area_slider = mo.ui.slider(start=0.0, stop=0.5, step=0.01, value=0.10, label="Min contour area ratio")
    padding_slider = mo.ui.slider(start=0, stop=30, step=1, value=5, label="Crop padding (px)")

    mo.vstack([
        mo.md("### Detection parameters"),
        mo.hstack([conf_slider, iou_slider]),
        mo.md("### Contour parameters"),
        mo.hstack([canny_low_slider, canny_high_slider]),
        mo.hstack([contour_area_slider, padding_slider]),
    ])
    return (
        canny_high_slider,
        canny_low_slider,
        conf_slider,
        contour_area_slider,
        iou_slider,
        padding_slider,
    )


@app.cell
def _(cv2, frame_slider, np, video_dropdown):
    _cap = cv2.VideoCapture(video_dropdown.value)
    _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider.value)
    _ret, raw_frame = _cap.read()
    _cap.release()

    def frame_to_png_bytes(frame, max_width=960):
        _h, _w = frame.shape[:2]
        if _w > max_width:
            _scale = max_width / _w
            frame = cv2.resize(frame, (max_width, int(_h * _scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    raw_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) if raw_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    return frame_to_png_bytes, raw_frame


@app.cell
def _():
    import anywidget
    import traitlets
    import base64

    class ClickableImage(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            canvas.style.maxWidth = '100%';
            canvas.style.cursor = 'crosshair';

            img.onload = () => {
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                redraw();
            };
            img.src = model.get('src');

            function redraw() {
                ctx.drawImage(img, 0, 0);
                const points = model.get('points');
                points.forEach(([x, y], i) => {
                    ctx.beginPath();
                    ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    ctx.fillStyle = 'red';
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 14px sans-serif';
                    ctx.fillText(i === 0 ? 'A' : 'B', x + 12, y + 5);
                });
                if (points.length === 2) {
                    ctx.beginPath();
                    ctx.moveTo(points[0][0], points[0][1]);
                    ctx.lineTo(points[1][0], points[1][1]);
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([8, 4]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }

            canvas.addEventListener('click', (e) => {
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);

                let points = model.get('points');
                if (points.length >= 2) {
                    points = [[x, y]];
                } else {
                    points = [...points, [x, y]];
                }
                model.set('points', points);
                model.save_changes();
                redraw();
            });

            model.on('change:src', () => {
                img.src = model.get('src');
            });
            model.on('change:points', redraw);

            el.appendChild(canvas);
        }
        export default { render };
        """
        src = traitlets.Unicode("").tag(sync=True)
        points = traitlets.List([]).tag(sync=True)

    return ClickableImage, base64


@app.cell
def _(ClickableImage, base64, cv2, mo, raw_frame):
    _, _buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _data_url = "data:image/jpeg;base64," + base64.b64encode(_buf.tobytes()).decode()

    calibration_widget = mo.ui.anywidget(ClickableImage(src=_data_url, points=[]))
    ref_length_cm = mo.ui.number(start=0.1, stop=500.0, step=0.1, value=10.0, label="Real-world distance (cm)")

    mo.vstack([
        mo.md("### Calibration — click two points on a reference object"),
        mo.md("Click **Point A**, then **Point B** on the ends of a known-length object. Click again to reset."),
        calibration_widget,
        ref_length_cm,
    ])
    return calibration_widget, ref_length_cm


@app.cell
def _(calibration_widget, mo, np, ref_length_cm):
    _points = calibration_widget.value.get("points", [])

    if len(_points) == 2:
        _p1, _p2 = _points
        _px_dist = float(np.sqrt((_p2[0] - _p1[0]) ** 2 + (_p2[1] - _p1[1]) ** 2))
        px_per_cm = _px_dist / ref_length_cm.value if ref_length_cm.value > 0 else 1.0
        mo.md(
            f"**A** ({_p1[0]}, {_p1[1]}) → **B** ({_p2[0]}, {_p2[1]}): "
            f"**{_px_dist:.1f} px** = **{ref_length_cm.value:.1f} cm** → **{px_per_cm:.2f} px/cm**"
        )
    else:
        px_per_cm = 1.0
        mo.md("*Click two points on the image to calibrate.*")
    return (px_per_cm,)


@app.cell
def _(
    canny_high_slider,
    canny_low_slider,
    conf_slider,
    contour_area_slider,
    cv2,
    device,
    iou_slider,
    letterbox,
    model,
    non_max_suppression,
    np,
    padding_slider,
    raw_frame,
    scale_coords,
    torch,
):
    def measure_fish_contour(frame, bbox, padding, canny_low, canny_high, min_contour_ratio):
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

    def detect_frame(frame, conf_thres, iou_thres, padding, canny_low, canny_high, min_contour_ratio, img_size=640):
        img = letterbox(frame, img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != "cpu" else img.float()
        img /= 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    bbox = (x1, y1, x2, y2)
                    m = measure_fish_contour(frame, bbox, padding, canny_low, canny_high, min_contour_ratio)
                    detections.append({
                        "bbox": bbox,
                        "confidence": float(conf),
                        "class": int(cls),
                        "length_px": m["length_px"],
                        "width_px": m["width_px"],
                        "angle": m["angle"],
                        "contour_found": m["contour_found"],
                        "rotated_rect": m["rotated_rect"],
                        "edges": m["edges"],
                    })
        return detections

    detections = detect_frame(
        raw_frame,
        conf_thres=conf_slider.value,
        iou_thres=iou_slider.value,
        padding=int(padding_slider.value),
        canny_low=int(canny_low_slider.value),
        canny_high=int(canny_high_slider.value),
        min_contour_ratio=contour_area_slider.value,
    )
    return detect_frame, detections


@app.cell
def _(cv2, detections, frame_to_png_bytes, mo, px_per_cm, raw_frame):
    _annotated = raw_frame.copy()
    for _det in detections:
        _x1, _y1, _x2, _y2 = _det["bbox"]
        _conf = _det["confidence"]
        _length_px = _det["length_px"]
        _width_px = _det["width_px"]
        _length_cm = _length_px / px_per_cm
        _width_cm = _width_px / px_per_cm
        _rotated_rect = _det["rotated_rect"]

        cv2.rectangle(_annotated, (_x1, _y1), (_x2, _y2), (0, 255, 0), 2)
        if _rotated_rect is not None:
            _box_points = cv2.boxPoints(_rotated_rect).astype(int)
            cv2.drawContours(_annotated, [_box_points], 0, (255, 255, 0), 2)

        _method = "" if _det["contour_found"] else " (bbox)"
        _label = f"Fish {_conf:.2f} | L:{_length_cm:.1f}cm W:{_width_cm:.1f}cm{_method}"
        (_tw, _th), _ = cv2.getTextSize(_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(_annotated, (_x1, _y1 - _th - 8), (_x1 + _tw, _y1), (0, 255, 0), -1)
        cv2.putText(_annotated, _label, (_x1, _y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    mo.hstack([
        mo.vstack([
            mo.md(f"### Detections ({len(detections)} fish)"),
            mo.image(frame_to_png_bytes(_annotated)),
        ]),
    ])
    return


@app.cell
def _(cv2, detections, mo, plt, raw_frame):
    _fish_with_edges = [(_i, _d) for _i, _d in enumerate(detections) if _d.get("edges") is not None]

    if _fish_with_edges:
        _n = len(_fish_with_edges)
        _fig, _axes = plt.subplots(2, _n, figsize=(4 * _n, 6), squeeze=False)

        for _col, (_i, _det) in enumerate(_fish_with_edges):
            _x1, _y1, _x2, _y2 = _det["bbox"]
            _pad = 5
            _h, _w = raw_frame.shape[:2]
            _crop = raw_frame[max(0, _y1 - _pad):min(_h, _y2 + _pad),
                              max(0, _x1 - _pad):min(_w, _x2 + _pad)]
            _crop_rgb = cv2.cvtColor(_crop, cv2.COLOR_BGR2RGB)

            _axes[0, _col].imshow(_crop_rgb)
            _axes[0, _col].set_title(f"Fish {_i + 1} crop")
            _axes[0, _col].axis("off")

            _axes[1, _col].imshow(_det["edges"], cmap="gray")
            _axes[1, _col].set_title(f"Canny edges")
            _axes[1, _col].axis("off")

        _fig.tight_layout()
        mo.vstack([mo.md("### Edge detection detail"), _fig])
    else:
        mo.md("*No contour edges to display — try adjusting Canny thresholds.*")
    return


@app.cell
def _(detections, mo, pd, px_per_cm):
    if detections:
        _rows = []
        for _i, _d in enumerate(detections):
            _rows.append({
                "Fish": _i + 1,
                "Confidence": round(_d["confidence"], 3),
                "Length (px)": round(_d["length_px"], 1),
                "Width (px)": round(_d["width_px"], 1),
                "Length (cm)": round(_d["length_px"] / px_per_cm, 1),
                "Width (cm)": round(_d["width_px"] / px_per_cm, 1),
                "Angle": round(_d["angle"], 1),
                "Method": "contour" if _d["contour_found"] else "bbox",
            })
        _df = pd.DataFrame(_rows)
        mo.vstack([
            mo.md(f"### Detection table (calibration: {px_per_cm:.2f} px/cm)"),
            mo.ui.table(_df),
        ])
    else:
        mo.md("*No detections on this frame.*")
    return


@app.cell
def _(
    canny_high_slider,
    canny_low_slider,
    conf_slider,
    contour_area_slider,
    cv2,
    detect_frame,
    iou_slider,
    mo,
    np,
    padding_slider,
    pd,
    plt,
    total_frames,
    video_dropdown,
):
    mo.md("### Multi-frame analysis")

    _sample_count = min(20, total_frames)
    _frame_indices = np.linspace(0, total_frames - 1, _sample_count, dtype=int)

    _cap = cv2.VideoCapture(video_dropdown.value)
    _all_detections = []
    for _idx in _frame_indices:
        _cap.set(cv2.CAP_PROP_POS_FRAMES, int(_idx))
        _ret, _frame = _cap.read()
        if not _ret:
            continue
        _dets = detect_frame(
            _frame,
            conf_thres=conf_slider.value,
            iou_thres=iou_slider.value,
            padding=int(padding_slider.value),
            canny_low=int(canny_low_slider.value),
            canny_high=int(canny_high_slider.value),
            min_contour_ratio=contour_area_slider.value,
        )
        for _d in _dets:
            _all_detections.append({
                "frame": int(_idx),
                "confidence": _d["confidence"],
                "length_px": _d["length_px"],
                "width_px": _d["width_px"],
                "method": "contour" if _d["contour_found"] else "bbox",
            })
    _cap.release()

    if _all_detections:
        _df = pd.DataFrame(_all_detections)

        _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

        _counts = _df.groupby("frame").size()
        _axes[0].bar(_counts.index, _counts.values, width=max(1, total_frames // 50))
        _axes[0].set_xlabel("Frame")
        _axes[0].set_ylabel("Fish count")
        _axes[0].set_title("Detections per frame")

        _axes[1].hist(_df["length_px"], bins=20, edgecolor="black")
        _axes[1].set_xlabel("Length (px)")
        _axes[1].set_ylabel("Count")
        _axes[1].set_title("Fish length distribution")

        _axes[2].scatter(_df["length_px"], _df["width_px"], alpha=0.5, s=15)
        _axes[2].set_xlabel("Length (px)")
        _axes[2].set_ylabel("Width (px)")
        _axes[2].set_title("Length vs Width")

        _fig.tight_layout()
        mo.vstack([
            mo.md(f"Sampled **{_sample_count}** frames, found **{len(_all_detections)}** total detections."),
            _fig,
        ])
    else:
        mo.md("*No detections found across sampled frames.*")
    return


if __name__ == "__main__":
    app.run()

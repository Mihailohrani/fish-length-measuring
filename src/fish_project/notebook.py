import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import warnings

    import torch

    warnings.filterwarnings(
        "ignore",
        message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.",
        category=UserWarning,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Fish Detection & Measurement

    Choose one image, review the detections, and optionally enable advanced
    measurement tools for calibration and size comparison.
    """)
    return


@app.cell
def _():
    import base64

    import anywidget
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import traitlets

    from fish_project import detect_frame as detect_frame_shared
    from fish_project import draw_detections, load_model
    from fish_project.paths import (
        ensure_local_data_dirs,
        get_default_image_browser_dir,
    )

    ensure_local_data_dirs()
    return (
        anywidget,
        base64,
        cv2,
        detect_frame_shared,
        draw_detections,
        get_default_image_browser_dir,
        load_model,
        np,
        pd,
        plt,
        traitlets,
    )


@app.cell
def _(anywidget, traitlets):
    class ClickableImage(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
            const wrap = document.createElement('div');
            wrap.style.position = 'relative';
            wrap.style.display = 'inline-block';
            wrap.style.maxWidth = '100%';

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            canvas.style.maxWidth = '100%';
            canvas.style.cursor = 'crosshair';
            canvas.style.display = 'block';

            // Zoom buttons
            const controls = document.createElement('div');
            controls.style.cssText = 'position:absolute;top:8px;right:8px;display:flex;gap:4px;z-index:10';
            const btnStyle = 'width:32px;height:32px;border:none;border-radius:4px;background:rgba(0,0,0,0.6);color:white;font:bold 18px sans-serif;cursor:pointer;display:flex;align-items:center;justify-content:center';
            const btnPlus = document.createElement('button');
            btnPlus.textContent = '+'; btnPlus.style.cssText = btnStyle;
            const btnMinus = document.createElement('button');
            btnMinus.textContent = '−'; btnMinus.style.cssText = btnStyle;
            const btnReset = document.createElement('button');
            btnReset.textContent = '⌂'; btnReset.style.cssText = btnStyle + ';font-size:16px';
            btnReset.title = 'Reset zoom';
            controls.append(btnPlus, btnMinus, btnReset);

            let zoom = 1, panX = 0, panY = 0;
            let dragging = false, dragX = 0, dragY = 0, panX0 = 0, panY0 = 0;

            img.onload = () => {
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                redraw();
            };
            img.src = model.get('src');

            function toImage(e) {
                const rect = canvas.getBoundingClientRect();
                const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
                const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
                return { sx, sy, ix: Math.round((sx - panX) / zoom), iy: Math.round((sy - panY) / zoom) };
            }

            function applyZoom(factor) {
                const prev = zoom;
                zoom = Math.max(0.5, Math.min(20, zoom * factor));
                const cx = canvas.width / 2, cy = canvas.height / 2;
                panX = cx - (cx - panX) * (zoom / prev);
                panY = cy - (cy - panY) * (zoom / prev);
                redraw();
            }

            function redraw() {
                ctx.setTransform(1, 0, 0, 1, 0, 0);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.translate(panX, panY);
                ctx.scale(zoom, zoom);

                ctx.drawImage(img, 0, 0);

                const points = model.get('points');
                const s = Math.max(img.naturalHeight / 1200, 1);
                const r = 8 * s / zoom, lw = 2 * s / zoom, fs = Math.round(14 * s / zoom);
                points.forEach(([x, y], i) => {
                    ctx.beginPath();
                    ctx.arc(x, y, r, 0, 2 * Math.PI);
                    ctx.fillStyle = 'red';
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = lw;
                    ctx.stroke();
                    ctx.fillStyle = 'white';
                    ctx.font = `bold ${fs}px sans-serif`;
                    ctx.fillText(i === 0 ? 'A' : 'B', x + 12 * s / zoom, y + 5 * s / zoom);
                });
                if (points.length === 2) {
                    ctx.beginPath();
                    ctx.moveTo(points[0][0], points[0][1]);
                    ctx.lineTo(points[1][0], points[1][1]);
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = lw;
                    ctx.setLineDash([8 * s / zoom, 4 * s / zoom]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                ctx.setTransform(1, 0, 0, 1, 0, 0);
                if (zoom !== 1) {
                    const label = zoom.toFixed(1) + 'x';
                    ctx.font = 'bold 14px sans-serif';
                    const tw = ctx.measureText(label).width;
                    ctx.fillStyle = 'rgba(0,0,0,0.55)';
                    ctx.fillRect(8, 8, tw + 16, 28);
                    ctx.fillStyle = 'white';
                    ctx.fillText(label, 16, 27);
                }
            }

            btnPlus.addEventListener('click', (e) => { e.stopPropagation(); applyZoom(1.3); });
            btnMinus.addEventListener('click', (e) => { e.stopPropagation(); applyZoom(1 / 1.3); });
            btnReset.addEventListener('click', (e) => { e.stopPropagation(); zoom = 1; panX = 0; panY = 0; redraw(); });

            canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const { sx, sy } = toImage(e);
                const prev = zoom;
                zoom = Math.max(0.5, Math.min(20, zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
                panX = sx - (sx - panX) * (zoom / prev);
                panY = sy - (sy - panY) * (zoom / prev);
                redraw();
            }, { passive: false });

            canvas.addEventListener('mousedown', (e) => {
                const { sx, sy } = toImage(e);
                dragX = sx; dragY = sy;
                panX0 = panX; panY0 = panY;
                dragging = false;
            });

            canvas.addEventListener('mousemove', (e) => {
                if (e.buttons !== 1) return;
                const { sx, sy } = toImage(e);
                if (Math.abs(sx - dragX) > 3 || Math.abs(sy - dragY) > 3) dragging = true;
                if (dragging) {
                    canvas.style.cursor = 'grabbing';
                    panX = panX0 + (sx - dragX);
                    panY = panY0 + (sy - dragY);
                    redraw();
                }
            });

            canvas.addEventListener('mouseup', (e) => {
                canvas.style.cursor = 'crosshair';
                if (dragging) { dragging = false; return; }
                const { ix, iy } = toImage(e);
                let points = model.get('points');
                points = points.length >= 2 ? [[ix, iy]] : [...points, [ix, iy]];
                model.set('points', points);
                model.save_changes();
                redraw();
            });

            model.on('change:src', () => {
                zoom = 1; panX = 0; panY = 0;
                img.src = model.get('src');
            });
            model.on('change:points', redraw);

            wrap.append(canvas, controls);
            el.appendChild(wrap);
        }
        export default { render };
        """

        src = traitlets.Unicode("").tag(sync=True)
        points = traitlets.List([]).tag(sync=True)

    return (ClickableImage,)


@app.cell
def _(mo):
    from fish_project.paths import detector_weight_choices

    _choices = detector_weight_choices()
    mo.stop(
        not _choices,
        mo.md(
            "*No detector weights found. Under `data/models/` add at least one of:*\n\n"
            "- `FishYolov7_tiny_ultralytics.onnx`\n"
            "- `fish-yolo26n.pt`"
        ),
    )
    weight_selector = mo.ui.dropdown(
        options=_choices,
        value=next(iter(_choices.values())),
        label="Detector model",
    )
    mo.vstack([mo.md("## Detector"), weight_selector])
    return (weight_selector,)


@app.cell
def _(load_model, mo, weight_selector):
    import torch
    from pathlib import Path

    _weights = Path(weight_selector.value)
    mo.stop(
        not _weights.is_file(),
        mo.md(f"*Model weights not found at `{_weights}`.*"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(_weights), device)
    print(f"Loaded `{_weights.name}` on {device}")
    return device, model


@app.cell
def _(base64, cv2, pd):
    def frame_to_image_bytes(frame, max_width=960):
        _height, _width = frame.shape[:2]
        if _width > max_width:
            _scale = max_width / _width
            frame = cv2.resize(frame, (max_width, int(_height * _scale)))
        _, _buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return _buffer.tobytes()

    def frame_to_data_url(frame):
        _, _buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return "data:image/jpeg;base64," + base64.b64encode(_buffer.tobytes()).decode()

    def build_detection_table(detections, px_per_cm=None):
        _rows = []
        for _index, _detection in enumerate(detections):
            _row = {
                "Fish": _index + 1,
                "Confidence": round(_detection["confidence"], 3),
                "Length (px)": round(_detection["length_px"], 1),
                "Width (px)": round(_detection["width_px"], 1),
                "Angle": round(_detection["angle"], 1),
                "Method": "contour" if _detection["contour_found"] else "bbox",
            }
            if px_per_cm is not None and px_per_cm > 0:
                _row["Length (cm)"] = round(_detection["length_px"] / px_per_cm, 1)
                _row["Width (cm)"] = round(_detection["width_px"] / px_per_cm, 1)
            _rows.append(_row)
        return pd.DataFrame(_rows)

    return build_detection_table, frame_to_data_url, frame_to_image_bytes


@app.cell
def _(get_default_image_browser_dir, mo):
    image_source = mo.ui.dropdown(
        options={"Sample library": "sample", "Upload": "upload"},
        value="Sample library",
        label="Image source",
    )
    sample_browser = mo.ui.file_browser(
        initial_path=get_default_image_browser_dir(),
        filetypes=[".jpg", ".jpeg", ".png", ".bmp"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Sample image",
    )
    image_upload = mo.ui.file(
        filetypes=[".jpg", ".jpeg", ".png", ".bmp"],
        label="Upload image",
    )
    return image_source, image_upload, sample_browser


@app.cell
def _(
    cv2,
    frame_to_image_bytes,
    get_default_image_browser_dir,
    image_source,
    image_upload,
    mo,
    np,
    sample_browser,
):
    _sample_root = get_default_image_browser_dir()
    _active_control = sample_browser if image_source.value == "sample" else image_upload
    _source_note = (
        mo.md(f"Sample library root: `{_sample_root}`")
        if image_source.value == "sample"
        else mo.md("Upload a single image from your machine.")
    )

    _picker = mo.vstack(
        [
            mo.md("## Choose Image"),
            image_source,
            _source_note,
            _active_control,
        ]
    )

    if image_source.value == "sample":
        mo.stop(
            not sample_browser.value,
            _picker,
        )
        _sample_file = sample_browser.value[0]
        _sample_path = _sample_file.path
        selected_image = cv2.imread(str(_sample_path))
        selected_image_name = _sample_path.name
        selected_image_origin = f"Sample library ({_sample_path.parent.name})"
    else:
        mo.stop(not image_upload.value, _picker)
        _uploaded_file = image_upload.value[0]
        _image_buffer = np.frombuffer(_uploaded_file.contents, np.uint8)
        selected_image = cv2.imdecode(_image_buffer, cv2.IMREAD_COLOR)
        selected_image_name = _uploaded_file.name
        selected_image_origin = "Upload"

    mo.stop(selected_image is None, _picker)

    mo.hstack(
        [
            _picker,
            mo.vstack(
                [
                    mo.md("## Selected Image"),
                    mo.md(f"**{selected_image_name}** from **{selected_image_origin}**"),
                    mo.image(frame_to_image_bytes(selected_image)),
                ]
            ),
        ]
    )
    return selected_image, selected_image_name, selected_image_origin


@app.cell
def _(mo):
    conf_slider = mo.ui.slider(
        start=0.05,
        stop=0.95,
        step=0.05,
        value=0.25,
        label="Confidence threshold",
    )
    iou_slider = mo.ui.slider(
        start=0.1,
        stop=0.9,
        step=0.05,
        value=0.45,
        label="IoU threshold",
    )
    canny_low_slider = mo.ui.slider(
        start=10,
        stop=200,
        step=10,
        value=50,
        label="Canny low",
    )
    canny_high_slider = mo.ui.slider(
        start=50,
        stop=300,
        step=10,
        value=150,
        label="Canny high",
    )
    contour_area_slider = mo.ui.slider(
        start=0.0,
        stop=0.5,
        step=0.01,
        value=0.10,
        label="Min contour area ratio",
    )
    padding_slider = mo.ui.slider(
        start=0,
        stop=30,
        step=1,
        value=5,
        label="Crop padding (px)",
    )
    use_clahe_switch = mo.ui.switch(
        label="CLAHE (contrast equalization)",
        value=False,
    )
    clahe_clip_slider = mo.ui.slider(
        start=1.0,
        stop=8.0,
        step=0.5,
        value=2.0,
        label="CLAHE clip limit",
    )
    use_bilateral_switch = mo.ui.switch(
        label="Bilateral denoise",
        value=False,
    )
    bilateral_sigma_slider = mo.ui.slider(
        start=10,
        stop=150,
        step=5,
        value=50,
        label="Bilateral σ",
    )
    return (
        bilateral_sigma_slider,
        canny_high_slider,
        canny_low_slider,
        clahe_clip_slider,
        conf_slider,
        contour_area_slider,
        iou_slider,
        padding_slider,
        use_bilateral_switch,
        use_clahe_switch,
    )


@app.cell
def _(detect_frame_shared, device, model):
    def detect_frame(
        frame,
        conf_thres,
        iou_thres,
        padding,
        canny_low,
        canny_high,
        min_contour_ratio,
        img_size=640,
        use_clahe=False,
        clahe_clip=2.0,
        use_bilateral=False,
        bilateral_sigma=50,
    ):
        return detect_frame_shared(
            model,
            frame,
            device,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            padding=padding,
            canny_low=canny_low,
            canny_high=canny_high,
            min_contour_ratio=min_contour_ratio,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            use_bilateral=use_bilateral,
            bilateral_sigma=bilateral_sigma,
        )

    return (detect_frame,)


@app.cell
def _(
    bilateral_sigma_slider,
    canny_high_slider,
    canny_low_slider,
    clahe_clip_slider,
    conf_slider,
    contour_area_slider,
    detect_frame,
    iou_slider,
    padding_slider,
    selected_image,
    use_bilateral_switch,
    use_clahe_switch,
):
    detections = detect_frame(
        selected_image,
        conf_thres=conf_slider.value,
        iou_thres=iou_slider.value,
        padding=int(padding_slider.value),
        canny_low=int(canny_low_slider.value),
        canny_high=int(canny_high_slider.value),
        min_contour_ratio=contour_area_slider.value,
        use_clahe=use_clahe_switch.value,
        clahe_clip=clahe_clip_slider.value,
        use_bilateral=use_bilateral_switch.value,
        bilateral_sigma=int(bilateral_sigma_slider.value),
    )
    return (detections,)


@app.cell
def _(ClickableImage, frame_to_data_url, mo, selected_image):
    calibration_widget = mo.ui.anywidget(
        ClickableImage(src=frame_to_data_url(selected_image), points=[])
    )
    ref_length_cm = mo.ui.number(
        start=0.1,
        stop=500.0,
        step=0.1,
        value=10.0,
        label="Real-world distance (cm)",
    )

    mo.vstack(
        [
            mo.md("## Calibration"),
            mo.md(
                "Click **Point A**, then **Point B** on the image to calibrate from pixels to centimeters. Click again to reset."
            ),
            calibration_widget,
            ref_length_cm,
        ]
    )
    return calibration_widget, ref_length_cm


@app.cell
def _(calibration_widget, mo, np, ref_length_cm):
    px_per_cm = None

    _points = calibration_widget.value.get("points", [])
    if len(_points) == 2 and ref_length_cm.value > 0:
        _point_a, _point_b = _points
        _px_dist = float(
            np.sqrt(
                (_point_b[0] - _point_a[0]) ** 2
                + (_point_b[1] - _point_a[1]) ** 2
            )
        )
        px_per_cm = _px_dist / ref_length_cm.value
        mo.output.replace(
            mo.md(
                f"**A** ({_point_a[0]}, {_point_a[1]}) → **B** ({_point_b[0]}, {_point_b[1]}): "
                f"**{_px_dist:.1f} px** = **{ref_length_cm.value:.1f} cm** → **{px_per_cm:.2f} px/cm**"
            )
        )
    else:
        mo.output.replace(
            mo.md("*Click two reference points on the image to calibrate.*")
        )
    return (px_per_cm,)


@app.cell
def _(
    bilateral_sigma_slider,
    canny_high_slider,
    canny_low_slider,
    clahe_clip_slider,
    conf_slider,
    contour_area_slider,
    detections,
    draw_detections,
    frame_to_image_bytes,
    iou_slider,
    mo,
    padding_slider,
    px_per_cm,
    selected_image,
    selected_image_name,
    selected_image_origin,
    use_bilateral_switch,
    use_clahe_switch,
):
    _annotated = draw_detections(
        selected_image.copy(),
        detections,
        px_per_unit=px_per_cm,
        unit_name="cm" if px_per_cm is not None else "px",
    )
    _unit_text = "cm" if px_per_cm is not None else "px"

    mo.vstack(
        [
            mo.md("## Result"),
            mo.md(f"**{selected_image_name}** from **{selected_image_origin}**"),
            mo.md(
                f"Detected **{len(detections)}** fish. Labels are shown in **{_unit_text}**."
            ),
            mo.image(frame_to_image_bytes(_annotated)),
            mo.md("## Detection Settings"),
            mo.hstack([conf_slider, iou_slider]),
            mo.hstack([canny_low_slider, canny_high_slider]),
            mo.hstack([contour_area_slider, padding_slider]),
            mo.md("## Preprocessing"),
            mo.hstack([use_clahe_switch, clahe_clip_slider]),
            mo.hstack([use_bilateral_switch, bilateral_sigma_slider]),
        ]
    )
    return


@app.cell
def _(build_detection_table, detections, mo, px_per_cm):
    if detections:
        _df = build_detection_table(detections, px_per_cm=px_per_cm)
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("## Detection Table"),
                    mo.ui.table(_df),
                ]
            )
        )
    else:
        mo.output.replace(mo.md("*No detections on this image.*"))
    return


@app.cell
def _(cv2, detections, mo, padding_slider, plt, selected_image):
    _fish_with_edges = [
        (_index, _detection)
        for _index, _detection in enumerate(detections)
        if _detection.get("edges") is not None
    ]

    if _fish_with_edges:
        _count = len(_fish_with_edges)
        _fig, _axes = plt.subplots(
            2,
            _count,
            figsize=(4 * _count, 6),
            squeeze=False,
        )

        for _col, (_index, _det) in enumerate(_fish_with_edges):
            _x1, _y1, _x2, _y2 = _det["bbox"]
            _pad = int(padding_slider.value)
            _height, _width = selected_image.shape[:2]
            _crop = selected_image[
                max(0, _y1 - _pad) : min(_height, _y2 + _pad),
                max(0, _x1 - _pad) : min(_width, _x2 + _pad),
            ]
            _crop_rgb = cv2.cvtColor(_crop, cv2.COLOR_BGR2RGB)

            _axes[0, _col].imshow(_crop_rgb)
            _axes[0, _col].set_title(f"Fish {_index + 1} crop")
            _axes[0, _col].axis("off")

            _axes[1, _col].imshow(_det["edges"], cmap="gray")
            _axes[1, _col].set_title("Canny edges")
            _axes[1, _col].axis("off")

        _fig.tight_layout()
        mo.vstack([mo.md("## Edge Detail"), _fig])
    else:
        mo.md("*No contour edges to display for this image.*")
    return


@app.cell
def _(detections, mo):
    if detections:
        actual_length_inputs = mo.ui.array(
            [
                mo.ui.number(
                    start=0,
                    stop=500,
                    step=0.1,
                    value=65,
                    label=f"Fish {index + 1} length (cm)",
                )
                for index in range(len(detections))
            ]
        )
        actual_width_inputs = mo.ui.array(
            [
                mo.ui.number(
                    start=0,
                    stop=500,
                    step=0.1,
                    value=23.5,
                    label=f"Fish {index + 1} width (cm)",
                )
                for index in range(len(detections))
            ]
        )

        mo.output.replace(
            mo.vstack(
                [
                    mo.md("## Actual Size Inputs"),
                    mo.md("Leave values at 0 to skip comparison for that fish."),
                    mo.hstack(
                        [
                            mo.vstack([mo.md("**Lengths**"), actual_length_inputs]),
                            mo.vstack([mo.md("**Widths**"), actual_width_inputs]),
                        ]
                    ),
                ]
            )
        )
    else:
        actual_length_inputs = None
        actual_width_inputs = None
        mo.output.replace(mo.md("*No detections to compare.*"))
    return actual_length_inputs, actual_width_inputs


@app.cell
def _(
    actual_length_inputs,
    actual_width_inputs,
    detections,
    mo,
    pd,
    px_per_cm,
):
    mo.stop(
        actual_length_inputs is None or actual_width_inputs is None,
        mo.md("*No detections — nothing to compare.*"),
    )
    mo.stop(
        px_per_cm is None,
        mo.md("*Calibrate the image first to enable size comparison.*"),
    )

    _rows = []
    for _index, _detection in enumerate(detections):
        _det_length = _detection["length_px"] / px_per_cm
        _det_width = _detection["width_px"] / px_per_cm
        _actual_length = actual_length_inputs.value[_index]
        _actual_width = actual_width_inputs.value[_index]

        _length_error = (
            abs(_det_length - _actual_length) if _actual_length > 0 else None
        )
        _length_error_pct = (
            _length_error / _actual_length * 100 if _actual_length > 0 else None
        )
        _width_error = (
            abs(_det_width - _actual_width) if _actual_width > 0 else None
        )
        _width_error_pct = (
            _width_error / _actual_width * 100 if _actual_width > 0 else None
        )

        _rows.append(
            {
                "Fish": _index + 1,
                "Det. L (cm)": round(_det_length, 1),
                "Act. L (cm)": round(_actual_length, 1)
                if _actual_length > 0
                else "—",
                "L error (cm)": round(_length_error, 2)
                if _length_error is not None
                else "—",
                "L error (%)": round(_length_error_pct, 1)
                if _length_error_pct is not None
                else "—",
                "Det. W (cm)": round(_det_width, 1),
                "Act. W (cm)": round(_actual_width, 1)
                if _actual_width > 0
                else "—",
                "W error (cm)": round(_width_error, 2)
                if _width_error is not None
                else "—",
                "W error (%)": round(_width_error_pct, 1)
                if _width_error_pct is not None
                else "—",
                "Method": "contour" if _detection["contour_found"] else "bbox",
            }
        )

    _df = pd.DataFrame(_rows)
    _with_actual = [
        _row for _row in _rows if isinstance(_row.get("L error (%)"), float)
    ]
    _summary = (
        f"Mean length error: **{sum(_row['L error (%)'] for _row in _with_actual) / len(_with_actual):.1f}%** "
        f"across {len(_with_actual)} fish with known sizes."
        if _with_actual
        else "*Enter actual sizes above to see error metrics.*"
    )

    mo.vstack(
        [
            mo.md("## Size Comparison"),
            mo.ui.table(_df),
            mo.md(_summary),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import warnings

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

    In **Result**, pick **YOLOv7** or **YOLO26** when both weights are installed.
    Then choose an image, review detections, and optionally use calibration and
    size comparison.
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
        BAGS_DIR,
        DOWNSAMPLED_IMAGES_DIR,
        ORIGINAL_IMAGES_DIR,
        VIDEOS_DIR,
        ensure_local_data_dirs,
    )

    ensure_local_data_dirs()
    return (
        BAGS_DIR,
        DOWNSAMPLED_IMAGES_DIR,
        ORIGINAL_IMAGES_DIR,
        VIDEOS_DIR,
        anywidget,
        base64,
        cv2,
        detect_frame_shared,
        draw_detections,
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
            btnReset.title = 'Reset zoom & pan';
            controls.append(btnPlus, btnMinus, btnReset);

            let zoom = 1, panX = 0, panY = 0;

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
                const rect = canvas.getBoundingClientRect();
                const s = Math.max(canvas.width / rect.width, 1);
                const r = 8 * s / zoom, lw = 3 * s / zoom, fs = Math.round(16 * s / zoom);
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

            function panBy(dx, dy) {
                panX += dx;
                panY += dy;
                redraw();
            }

            const panPad = document.createElement('div');
            panPad.style.cssText = 'position:absolute;bottom:8px;right:8px;display:flex;flex-direction:column;align-items:center;gap:4px;z-index:10';
            const panStyle = btnStyle.replace('32px', '36px') + ';font-size:16px;line-height:1';
            const mkPanBtn = (label, title, dx, dy) => {
                const b = document.createElement('button');
                b.type = 'button';
                b.textContent = label;
                b.style.cssText = panStyle;
                b.title = title;
                b.addEventListener('click', (ev) => { ev.stopPropagation(); panBy(dx, dy); });
                return b;
            };
            const PAN_STEP = 48;
            const panRow = document.createElement('div');
            panRow.style.cssText = 'display:flex;flex-direction:row;gap:4px';
            panRow.append(
                mkPanBtn('\\u2190', 'Pan left', PAN_STEP, 0),
                mkPanBtn('\\u2192', 'Pan right', -PAN_STEP, 0),
            );
            panPad.append(
                mkPanBtn('\\u2191', 'Pan up', 0, PAN_STEP),
                panRow,
                mkPanBtn('\\u2193', 'Pan down', 0, -PAN_STEP),
            );

            btnPlus.addEventListener('click', (e) => { e.stopPropagation(); applyZoom(1.3); });
            btnMinus.addEventListener('click', (e) => { e.stopPropagation(); applyZoom(1 / 1.3); });
            btnReset.addEventListener('click', (e) => { e.stopPropagation(); zoom = 1; panX = 0; panY = 0; redraw(); });

            canvas.addEventListener('click', (e) => {
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

            wrap.append(canvas, controls, panPad);
            el.appendChild(wrap);
        }
        export default { render };
        """

        src = traitlets.Unicode("").tag(sync=True)
        points = traitlets.List([]).tag(sync=True)

    return (ClickableImage,)


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
def _(BAGS_DIR, DOWNSAMPLED_IMAGES_DIR, ORIGINAL_IMAGES_DIR, VIDEOS_DIR, mo):
    image_source = mo.ui.dropdown(
        options={
            "Original": "original",
            "Downscaled": "downsampled",
            "Upload": "upload",
            "Video": "video",
            "Bag": "bag",
        },
        value="Original",
        label="Image source",
    )
    sample_browser_original = mo.ui.file_browser(
        initial_path=ORIGINAL_IMAGES_DIR,
        filetypes=[".jpg", ".jpeg", ".png", ".bmp"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Original image",
    )
    sample_browser_downscaled = mo.ui.file_browser(
        initial_path=DOWNSAMPLED_IMAGES_DIR,
        filetypes=[".jpg", ".jpeg", ".png", ".bmp"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Downscaled image",
    )
    image_upload = mo.ui.file(
        filetypes=[".jpg", ".jpeg", ".png", ".bmp"],
        label="Upload image",
    )
    video_browser = mo.ui.file_browser(
        initial_path=VIDEOS_DIR,
        filetypes=[".mp4", ".avi", ".mov"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Video file",
    )
    bag_browser = mo.ui.file_browser(
        initial_path=BAGS_DIR,
        filetypes=[".bag"],
        selection_mode="file",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Bag file",
    )
    return (
        bag_browser,
        image_source,
        image_upload,
        sample_browser_downscaled,
        sample_browser_original,
        video_browser,
    )


@app.cell
def _(bag_browser, cv2, image_source, mo, video_browser):
    frame_selector = None
    depth_only_bag = False
    bag_intrinsics = None

    if image_source.value == "video" and video_browser.value:
        _path = str(video_browser.value[0].path)
        _cap = cv2.VideoCapture(_path)
        _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _fps = _cap.get(cv2.CAP_PROP_FPS)
        _cap.release()
        if _total > 0:
            frame_selector = mo.ui.slider(
                start=0,
                stop=_total - 1,
                step=1,
                value=0,
                label=f"Frame (of {_total}, {_fps:.1f} fps)",
            )
    elif image_source.value == "bag" and bag_browser.value:
        from fish_project.bag_reader import count_bag_frames, has_color_stream

        _bag_path = str(bag_browser.value[0].path)
        _total = count_bag_frames(_bag_path)
        if _total > 0:
            frame_selector = mo.ui.slider(
                start=0,
                stop=_total - 1,
                step=1,
                value=0,
                label=f"Frame (of {_total})",
            )
        if not has_color_stream(_bag_path):
            depth_only_bag = True
            from fish_project.bag_reader import get_depth_intrinsics

            bag_intrinsics = get_depth_intrinsics(_bag_path)
    return bag_intrinsics, depth_only_bag, frame_selector


@app.cell
def _(
    BAGS_DIR,
    DOWNSAMPLED_IMAGES_DIR,
    ORIGINAL_IMAGES_DIR,
    VIDEOS_DIR,
    bag_browser,
    cv2,
    frame_selector,
    frame_to_image_bytes,
    image_source,
    image_upload,
    mo,
    np,
    sample_browser_downscaled,
    sample_browser_original,
    video_browser,
):
    if image_source.value == "original":
        _active_control = sample_browser_original
        _source_note = mo.md(f"Library root: `{ORIGINAL_IMAGES_DIR}`")
    elif image_source.value == "downsampled":
        _active_control = sample_browser_downscaled
        _source_note = mo.md(f"Library root: `{DOWNSAMPLED_IMAGES_DIR}`")
    elif image_source.value == "video":
        _active_control = video_browser
        _source_note = mo.md(f"Library root: `{VIDEOS_DIR}`")
    elif image_source.value == "bag":
        _active_control = bag_browser
        _source_note = mo.md(f"Library root: `{BAGS_DIR}`")
    else:
        _active_control = image_upload
        _source_note = mo.md("Upload a single image from your machine.")

    _picker_items = [
        mo.md("## Choose Image"),
        image_source,
        _source_note,
        _active_control,
    ]
    if frame_selector is not None and image_source.value in ("video", "bag"):
        _picker_items.append(frame_selector)
    _picker = mo.vstack(_picker_items)

    if image_source.value == "upload":
        mo.stop(not image_upload.value, _picker)
        _uploaded_file = image_upload.value[0]
        _image_buffer = np.frombuffer(_uploaded_file.contents, np.uint8)
        selected_image = cv2.imdecode(_image_buffer, cv2.IMREAD_COLOR)
        selected_image_name = _uploaded_file.name
        selected_image_origin = "Upload"
    elif image_source.value == "video":
        mo.stop(not video_browser.value, _picker)
        mo.stop(frame_selector is None, _picker)
        _video_path = str(video_browser.value[0].path)
        _cap = cv2.VideoCapture(_video_path)
        _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_selector.value)
        _ret, selected_image = _cap.read()
        _cap.release()
        mo.stop(not _ret, _picker)
        selected_image_name = f"{video_browser.value[0].path.name} frame {frame_selector.value}"
        selected_image_origin = "Video"
    elif image_source.value == "bag":
        mo.stop(not bag_browser.value, _picker)
        mo.stop(frame_selector is None, _picker)
        from fish_project.bag_reader import extract_frame

        selected_image = extract_frame(
            str(bag_browser.value[0].path), frame_selector.value
        )
        mo.stop(selected_image is None, _picker)
        selected_image_name = f"{bag_browser.value[0].path.name} frame {frame_selector.value}"
        selected_image_origin = "Bag"
    else:
        _browser = (
            sample_browser_original
            if image_source.value == "original"
            else sample_browser_downscaled
        )
        mo.stop(
            not _browser.value,
            _picker,
        )
        _sample_file = _browser.value[0]
        _sample_path = _sample_file.path
        selected_image = cv2.imread(str(_sample_path))
        selected_image_name = _sample_path.name
        selected_image_origin = (
            "Original" if image_source.value == "original" else "Downscaled"
        )

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
def _(ClickableImage, depth_only_bag, frame_to_data_url, mo, selected_image):
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

    if depth_only_bag:
        _output = mo.vstack(
            [
                mo.md("## Distance Measurement"),
                mo.md(
                    "Click **Point A**, then **Point B** on the depth image to measure pixel distance. "
                    "Zoom with **+ / −**; use the **arrow buttons** to pan. Click again to reset points."
                ),
                calibration_widget,
            ]
        )
    else:
        _output = mo.vstack(
            [
                mo.md("## Calibration"),
                mo.md(
                    "Click **Point A**, then **Point B** on the image to calibrate from pixels to centimeters. "
                    "Zoom with **+ / −**; mouse wheel scrolls the page. When zoomed, use the **arrow buttons** to pan. Click again to reset points."
                ),
                calibration_widget,
                ref_length_cm,
            ]
        )
    _output
    return calibration_widget, ref_length_cm


@app.cell
def _(
    bag_browser,
    bag_intrinsics,
    calibration_widget,
    depth_only_bag,
    frame_selector,
    mo,
    np,
    ref_length_cm,
):
    px_per_cm = None

    _points = calibration_widget.value.get("points", [])

    if depth_only_bag:
        if len(_points) == 2:
            _point_a, _point_b = _points
            _result = f"**A** ({_point_a[0]}, {_point_a[1]}) → **B** ({_point_b[0]}, {_point_b[1]})"

            if bag_intrinsics and bag_browser.value and frame_selector is not None:
                from fish_project.bag_reader import extract_raw_depth

                _raw = extract_raw_depth(
                    str(bag_browser.value[0].path), frame_selector.value
                )
                if _raw is not None:
                    _d1 = float(_raw[_point_a[1], _point_a[0]])
                    _d2 = float(_raw[_point_b[1], _point_b[0]])
                    _du = bag_intrinsics["depth_units"]
                    _fx = bag_intrinsics["fx"]
                    _fy = bag_intrinsics["fy"]
                    _cx = bag_intrinsics["cx"]
                    _cy = bag_intrinsics["cy"]

                    _x1 = (_point_a[0] - _cx) * _d1 * _du / _fx
                    _y1 = (_point_a[1] - _cy) * _d1 * _du / _fy
                    _z1 = _d1 * _du
                    _x2 = (_point_b[0] - _cx) * _d2 * _du / _fx
                    _y2 = (_point_b[1] - _cy) * _d2 * _du / _fy
                    _z2 = _d2 * _du

                    _dist_m = float(
                        np.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2 + (_z2 - _z1) ** 2)
                    )
                    _dist_cm = _dist_m * 100
                    _result += f": **{_dist_cm:.1f} cm**"
                    if _d1 == 0 or _d2 == 0:
                        _result += " *(warning: one or both points have zero depth)*"

            mo.output.replace(mo.md(_result))
        else:
            mo.output.replace(
                mo.md("*Click two points on the depth image to measure distance.*")
            )
    else:
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
def _(depth_only_bag, mo):
    weight_selector = None
    mo.stop(depth_only_bag)

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
        value=next(iter(_choices.keys())),
        label="Detector model",
    )
    return (weight_selector,)


@app.cell
def _(
    bilateral_sigma_slider,
    canny_high_slider,
    canny_low_slider,
    clahe_clip_slider,
    conf_slider,
    contour_area_slider,
    depth_only_bag,
    detect_frame_shared,
    draw_detections,
    frame_to_image_bytes,
    iou_slider,
    load_model,
    mo,
    padding_slider,
    px_per_cm,
    selected_image,
    selected_image_name,
    selected_image_origin,
    use_bilateral_switch,
    use_clahe_switch,
    weight_selector,
):
    detections = []
    mo.stop(depth_only_bag)

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

    detections = detect_frame_shared(
        model,
        selected_image,
        device,
        img_size=640,
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
            mo.hstack(
                [
                    mo.md("**Detector model:**"),
                    weight_selector,
                ],
                align="center",
                gap=0.75,
            ),
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
    return (detections,)


@app.cell
def _(build_detection_table, depth_only_bag, detections, mo, px_per_cm):
    mo.stop(depth_only_bag)
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
def _(
    cv2,
    depth_only_bag,
    detections,
    mo,
    padding_slider,
    plt,
    selected_image,
):
    mo.stop(depth_only_bag)
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
        _edge_detail = mo.vstack([mo.md("## Edge Detail"), _fig])
    else:
        _edge_detail = mo.md("*No contour edges to display for this image.*")

    _edge_detail
    return


@app.cell
def _(depth_only_bag, detections, mo):
    actual_length_inputs = None
    actual_width_inputs = None
    mo.stop(depth_only_bag)
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
        mo.output.replace(mo.md("*No detections to compare.*"))
    return actual_length_inputs, actual_width_inputs


@app.cell
def _(
    actual_length_inputs,
    actual_width_inputs,
    depth_only_bag,
    detections,
    mo,
    pd,
    px_per_cm,
):
    mo.stop(depth_only_bag)
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

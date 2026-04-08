# Fish Length Measuring

Detect and measure fish from images using YOLO object detection and contour analysis, served through an
interactive [Marimo](https://marimo.io) notebook.

## Requirements

Only [mise](https://mise.jdx.dev/installing-mise.html#installation-methods) is required.
Everything else is installed automatically: mise installs [uv](https://docs.astral.sh/uv/),
uv installs Python and all dependencies.

## Quick Start

```bash
git clone https://github.com/Mihailohrani/fish-length-measuring.git
cd fish-length-measuring
mise trust            # trust this project's mise.toml
mise install          # installs uv + git-lfs
mise run setup        # installs Python deps, pulls LFS data (models, images)
mise run prod         # opens the notebook in your browser
```

`mise run dev` starts the notebook in edit mode with MCP support.

## Docker

```bash
docker build -t fish-length .
docker run --rm -p 2718:2718 fish-length
```

Open `http://localhost:2718`.

## How It Works

1. **Detection** — YOLO models (`YOLOv7` ONNX or `YOLO26` PyTorch) locate fish in the frame.
2. **Measurement** — Each bounding box is cropped, edge-detected (Canny), and the largest contour is fit with a
   minimum-area rotated rectangle to extract length and width in pixels.
3. **Calibration** — Click two reference points of known distance on the image to convert pixel measurements to
   centimeters.
4. **Comparison** — Enter actual fish sizes to see measurement error percentages.
5. **Depth measurement** — For depth-only `.bag` files (no color stream), click two points on the depth heat map
   to get a real-world distance in cm computed from camera intrinsics and raw depth values.

Supported input sources: still images (original, downsampled, or uploaded), video files (`.mp4`),
and ROS 1 bag files (`.bag`, e.g. Intel RealSense recordings). Video and bag sources include a
frame slider for scrubbing through frames.

## Project Structure

```
src/fish_project/
    notebook.py       Marimo notebook (UI)
    detection.py      YOLO inference + per-fish measurement
    measurement.py    Contour-based length/width extraction
    visualization.py  Bounding box and annotation drawing
    model.py          Model loading (Ultralytics)
    bag_reader.py     ROS 1 .bag frame extraction
    paths.py          Data directory conventions
data/
    models/           Detector weights (.onnx, .pt)    [LFS]
    images/           Sample fish images               [LFS]
    videos/           Sample videos                    [LFS]
    bags/             RealSense .bag recordings        [LFS]
```

## Mise Tasks

| Task             | Description                                 |
|------------------|---------------------------------------------|
| `mise run setup` | Install Python deps and pull Git LFS assets |
| `mise run dev`   | Launch notebook in edit mode                |
| `mise run prod`  | Launch notebook in read-only mode           |

# Fish Length Measuring

Detect and measure fish from images using YOLO object detection and contour analysis, served through an
interactive [Marimo](https://marimo.io) notebook.

## Quick Start

Requires [mise](https://mise.jdx.dev) (`curl https://mise.run | sh`).

```bash
git clone https://github.com/Mihailohrani/fish-length-measuring.git
cd fish-length-measuring
mise install          # installs uv + git-lfs
mise run setup        # installs Python deps, pulls LFS data (models, images)
mise run prod         # opens the notebook in your browser
```

`mise run dev` starts the notebook in edit mode with MCP support.

## Docker

```bash
docker build -t fish-length .
docker run -p 2718:2718 fish-length
```

Open `http://localhost:2718`.

## How It Works

1. **Detection** -- YOLO models (`YOLOv7` ONNX or `YOLO26` PyTorch) locate fish in the image.
2. **Measurement** -- Each bounding box is cropped, edge-detected (Canny), and the largest contour is fit with a
   minimum-area rotated rectangle to extract length and width in pixels.
3. **Calibration** -- Click two reference points of known distance on the image to convert pixel measurements to
   centimeters.
4. **Comparison** -- Enter actual fish sizes to see measurement error percentages.

## Project Structure

```
src/fish_project/
    notebook.py       Marimo notebook (UI)
    detection.py      YOLO inference + per-fish measurement
    measurement.py    Contour-based length/width extraction
    visualization.py  Bounding box and annotation drawing
    model.py          Model loading (Ultralytics)
    paths.py          Data directory conventions
data/
    models/           Detector weights (.onnx, .pt)  [LFS]
    images/           Sample fish images              [LFS]
    videos/           Sample videos                   [LFS]
```

## Mise Tasks

| Task             | Description                                 |
|------------------|---------------------------------------------|
| `mise run setup` | Install Python deps and pull Git LFS assets |
| `mise run dev`   | Launch notebook in edit mode                |
| `mise run prod`  | Launch notebook in read-only mode           |

## Requirements

- Python 3.14+
- Dependencies managed by [uv](https://docs.astral.sh/uv/) (installed automatically via mise)
- Binary assets tracked with [Git LFS](https://git-lfs.com) (installed automatically via mise)

# 🏀 Basketball Player Detection & Tracking Pipeline

> **Big Vision Internship Assignment** — End-to-end computer vision system for detecting and tracking basketball players and the ball using YOLOv11m + ByteTrack.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLOv11m-Ultralytics-purple?logo=yolo)
![Supervision](https://img.shields.io/badge/Supervision-Roboflow-orange)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 📋 Overview

This project implements a production-ready basketball tracking pipeline that:

- **Detects** players and the ball in 4K basketball footage using a fine-tuned **YOLOv11m** model
- **Tracks** persistent player identities across frames using **ByteTrack** (Kalman filter + two-stage matching)
- **Generates** spatial analytics — heatmaps, zone occupancy, and trajectory visualizations
- **Deploys** an interactive **Gradio** web demo for real-time video processing

---

## 🔗 Project Resources

📂 **Google Drive (Full Project Assets):** [Open in Drive](https://drive.google.com/drive/folders/1XeJBRmKvTCIiReYY5_Iixnz6TFS_oA-S?usp=sharing)

> Includes datasets, trained model weights, input videos, output results, and documentation.

---

## 🏗️ Project Structure

```
Big Vision/
├── Code/
│   ├── Final/                          # Production code
│   │   ├── basketball_LOCAL_FINAL.ipynb     # 🔑 Main notebook (local GPU)
│   │   ├── basketball_COLAB_FINAL.ipynb     # Google Colab version
│   │   ├── basketball_project/              # Generated at runtime
│   │   │   ├── datasets/                    # Downloaded Roboflow datasets
│   │   │   ├── merged_dataset/              # Cleaned & merged YOLO dataset
│   │   │   ├── extracted_frames/            # Sampled video frames
│   │   │   ├── outputs/                     # Tracked videos, heatmaps, metrics
│   │   │   ├── runs/                        # Training logs & checkpoints
│   │   │   └── weights/                     # Saved model weights
│   │   └── runs/                            # Ultralytics training runs
│   │       └── detect/yolov11m_basketball/
│   │           └── weights/best.pt          # ⭐ Trained model
│   └── Inputs/                         # Raw input videos (4K)
│       ├── 14789160_3840_2160_60fps.mp4
│       └── 14800461_3840_2160_60fps.mp4
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (tested on RTX 4050)
- **~8 GB** free disk space (datasets + model)

### 1. Install Dependencies

```bash
pip install ultralytics supervision gradio opencv-python numpy matplotlib roboflow
```

### 2. Run the Notebook

Open `Code/Final/basketball_LOCAL_FINAL.ipynb` in Jupyter and run cells sequentially:

| Cell | Purpose | Time |
|------|---------|------|
| 1 | Setup directories & detect GPU | ~5s |
| 2–5 | Download datasets from Roboflow | ~3 min |
| 6–7 | Merge & clean datasets (~5000 images) | ~2 min |
| **8** | **Train YOLOv11m** (100 epochs) | **~4 hours** |
| 9–12 | Validate model & extract frames | ~1 min |
| **13** | **Track video with ByteTrack** | **~10 min** |
| 14–16 | Generate heatmaps & analytics | ~2 min |
| **17** | **Launch Gradio web demo** | Instant |

> **Skip training:** If you already have `best.pt`, place it at `Code/Final/runs/detect/yolov11m_basketball/weights/best.pt` and skip to Cell 9.

### 3. Launch the Demo

```python
# Cell 17 launches the Gradio interface
demo.launch(share=False, inbrowser=True)
# → Opens at http://localhost:7860
```

---

## 🧠 Technical Architecture

### Detection — YOLOv11m

- **Base model**: `yolo11m.pt` (COCO pre-trained)
- **Fine-tuned** on ~5000 basketball images from 3 Roboflow datasets
- **Classes**: `Player` (0), `Ball` (1)
- **Inference**: `conf=0.40`, `iou=0.5`
- **Post-filter**: Aspect-ratio filter removes backboard/post false positives

### Tracking — ByteTrack

- **Algorithm**: Kalman filter prediction + two-stage Hungarian matching
- **High-confidence match** → immediate ID assignment
- **Low-confidence match** → tentative track, confirmed after persistence
- Handles occlusions and jersey color similarities in crowded scenes

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Epochs | 100 |
| Image Size | 640px |
| Batch Size | Auto |
| Augmentation | Ultralytics defaults |

### Results

| Metric | Score |
|--------|-------|
| **mAP@50** | **0.972** (97.2%) |
| mAP@50-95 | 0.65+ |
| Precision | 0.95+ |
| Recall | 0.94+ |

---

## 📊 Analytics & Visualizations

The pipeline generates the following outputs in `basketball_project/outputs/`:

| Output | Description |
|--------|-------------|
| `tracked_output.mp4` | Full video with bounding boxes, IDs, and trails |
| `detection_test.png` | Sample frame detections grid |
| `player_heatmap.png` | Gaussian-blurred player movement heatmap |
| `ball_heatmap.png` | Ball trajectory heatmap |
| `spatial_dashboard.png` | Zone occupancy (Paint, Mid-range, 3pt) |
| `tracking_metrics.json` | MOTA, ID switches, track lengths |

---

## 🖥️ Gradio Web Interface

The interactive demo allows uploading any basketball video and produces tracked output with:
- Real-time progress bar
- Bounding boxes with player IDs and confidence scores
- Movement trails via TraceAnnotator
- Frame-by-frame player count overlay

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥8.0 | YOLOv11 training & inference |
| `supervision` | ≥0.18 | Tracking, annotation, video I/O |
| `gradio` | ≥4.0 | Interactive web demo |
| `opencv-python` | ≥4.8 | Image/video processing |
| `roboflow` | ≥1.0 | Dataset download |
| `numpy` | ≥1.24 | Numerical operations |
| `matplotlib` | ≥3.7 | Visualization & plots |

---

## ⚠️ Known Issues & Fixes

| Issue | Solution |
|-------|----------|
| `ConnectionResetError [WinError 10054]` on Windows | Monkey-patch `asyncio.proactor_events` (included in Cell 17) |
| `ValueError: tracker_id missing` | Guard with `if dets.tracker_id is not None` before `TraceAnnotator` |
| `PermissionError` with Gradio uploads | Set `GRADIO_TEMP_DIR` to project directory |
| Video not playing in Gradio | Re-encode output to H.264 with ffmpeg |

---

## 👤 Author

**Mithun** — Big Vision Internship Assignment

---

## 📄 License

This project is developed as part of an academic internship assignment. All rights reserved.
